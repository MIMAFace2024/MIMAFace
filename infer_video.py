
import os
import sys
import cv2
import glob
import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as ttf
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from einops import rearrange
from kornia.geometry.transform import warp_affine
from accelerate.utils import set_seed
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda:0')

from transformers import AutoTokenizer, PretrainedConfig, CLIPVisionModel, CLIPImageProcessor, CLIPTokenizer
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL, EulerDiscreteScheduler

from dataset import face_util 
from pipeline import MIMAfacePipeline
from util import parse_args, save_videos_img2mp4, convert_batch_to_nprgb, unframe_mp42pil
from models.mimaface import MIMAFaceModel, MIMAFacePostfuseModule
from models.animatediff.unet import UNet3DConditionModel
from models.D3DFR.D3DFR_ReconModel import D3DFR_wrapper
from models.ArcFace.CurricularFace_wrapper import CurricularFaceOurCrop


torch.set_grad_enabled(False)
motion_module_kwargs_v3 = {"num_attention_heads" : 8,
                        "num_transformer_block" : 1,
                        "attention_block_types" : [ "Temporal_Self", "Temporal_Self" ],
                        "temporal_position_encoding" : True,
                        "temporal_position_encoding_max_len" : 32,
                        "temporal_attention_dim_div" : 1,
                        "zero_initialize" : True}

unet_additional_kwargs_v3 = {"use_inflated_groupnorm" : True,
                          "use_motion_module" : True,
                          "motion_module_resolutions" : [1,2,4,8],
                          "motion_module_mid_block" : False,
                          "motion_module_type" : "Vanilla",
                          "motion_module_kwargs" : motion_module_kwargs_v3}



def infer_one_video(src_img_path, drive_img_folder_path, video_length, save_path, overlap=3, clip=42):
    # process source
    src_im_pil = Image.open(src_img_path).convert("RGB")
    boxes, _, landmarks = mtcnn.detect(src_im_pil, landmarks=True)
    crop_source=True
    if crop_source:
        dets = boxes[0]
        # scaled box 
        crop_ratio = -1
        if crop_ratio>0:
            bbox = dets[0:4]
            bbox_size = max(bbox[2]-bbox[0], bbox[2]-bbox[0])
            bbox_x = 0.5*(bbox[2]+bbox[0])
            bbox_y = 0.5*(bbox[3]+bbox[1])
            x1 = bbox_x-bbox_size*crop_ratio
            x2 = bbox_x+bbox_size*crop_ratio
            y1 = bbox_y-bbox_size*crop_ratio
            y2 = bbox_y+bbox_size*crop_ratio
            bbox_pts4 = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], dtype=np.float32)   
        else:
        # original box
            bbox = dets[0:4].reshape((2,2))
            bbox_pts4 = face_util.get_box_lm4p(bbox)        

        warp_mat_crop = face_util.transformation_from_points(bbox_pts4, face_util.mean_box_lm4p_512)
        src_im_crop512 = cv2.warpAffine(np.array(src_im_pil), warp_mat_crop, (512, 512), flags=cv2.INTER_LINEAR)
        src_im_pil = Image.fromarray(src_im_crop512)

    _, _, landmarks = mtcnn.detect(src_im_pil, landmarks=True)
    pts5 = landmarks[0]
    image_src_warpmat256 = face_util.get_affine_transform(pts5, face_util.mean_face_lm5p_256)
    image_src_inverse_warpmat256 = cv2.invertAffineTransform(image_src_warpmat256)
    src_im_crop256 = cv2.warpAffine(np.array(src_im_pil), image_src_warpmat256, (256, 256), flags=cv2.INTER_LINEAR)
    images_src = pil2tensor(src_im_pil).view(1,3,test_image_size,test_image_size).to(device) 
    
    # process target
    drive_img_list = [x for x in os.listdir(drive_img_folder_path) if x.endswith('.jpg') or x.endswith('.png')]
    drive_img_list.sort()
    drive_img_length = min(len(drive_img_list), clip)
    
    latents = None
    latents_pre = None
    warp_mat_crop = None
    current_frame_start_index= 0
    while(current_frame_start_index < drive_img_length):
        images_tar_list = []
        image_tar_crop256_list = []
        image_tar_warpmat256_list = []
        image_tar_inverse_warpmat256_list = []
        # import pdb;pdb.set_trace()
        for i in range(current_frame_start_index, current_frame_start_index+video_length):
            i = min(i, drive_img_length-1)
            drive_img_path = os.path.join(drive_img_folder_path, drive_img_list[i])
            drive_im_pil = Image.open(drive_img_path).convert("RGB").resize((test_image_size,test_image_size))

            crop_drive=True
            if crop_drive:
                boxes, _, landmarks = mtcnn.detect(drive_im_pil, landmarks=True)
                if current_frame_start_index == 0 and i==0:
                    dets = boxes[0]
                    # scaled box 
                    crop_ratio = -1
                    if crop_ratio>0:
                        bbox = dets[0:4]
                        bbox_size = max(bbox[2]-bbox[0], bbox[2]-bbox[0])
                        bbox_x = 0.5*(bbox[2]+bbox[0])
                        bbox_y = 0.5*(bbox[3]+bbox[1])
                        x1 = bbox_x-bbox_size*crop_ratio
                        x2 = bbox_x+bbox_size*crop_ratio
                        y1 = bbox_y-bbox_size*crop_ratio
                        y2 = bbox_y+bbox_size*crop_ratio
                        bbox_pts4 = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], dtype=np.float32)   
                    else:
                    # original box
                        bbox = dets[0:4].reshape((2,2))
                        bbox_pts4 = face_util.get_box_lm4p(bbox)        

                    warp_mat_crop = face_util.transformation_from_points(bbox_pts4, face_util.mean_box_lm4p_512)   
                drive_im_crop512 = cv2.warpAffine(np.array(drive_im_pil), warp_mat_crop, (512, 512), flags=cv2.INTER_LINEAR)
                drive_im_pil = Image.fromarray(drive_im_crop512)
                
            boxes, _, landmarks = mtcnn.detect(drive_im_pil, landmarks=True)
            pts5 = landmarks[0]

            image_tar_warpmat256 = face_util.get_affine_transform(pts5, face_util.mean_face_lm5p_256)
            image_tar_warpmat256_list.append(image_tar_warpmat256)
            image_tar_inverse_warpmat256_list.append(cv2.invertAffineTransform(image_tar_warpmat256)) #####
            
            drive_im_crop256 = cv2.warpAffine(np.array(drive_im_pil), image_tar_warpmat256, (256, 256), flags=cv2.INTER_LINEAR)
            images_tar = pil2tensor(drive_im_pil).view(1,3,test_image_size,test_image_size).to(device) 
            # ======
            images_tar_list.append(images_tar)


        images_tar = torch.cat(images_tar_list, dim=0)
        image_tar_warpmat256 = torch.tensor(np.stack(image_tar_warpmat256_list, axis=0)).unsqueeze(0).to(device).to(dtype=weight_dtype) 
        image_tar_inverse_warpmat256 = torch.tensor(np.stack(image_tar_inverse_warpmat256_list, axis=0)).unsqueeze(0).to(device).to(dtype=weight_dtype) ###### (10, 2, 3)
        
        # D3DFR_inst
        src_img = images_src # torch.Size([2, 3, 512, 512])
        drv_same_img = rearrange(images_tar.unsqueeze(0), "b f c h w -> (b f) c h w")
        src_wapmat = torch.tensor(image_src_warpmat256).unsqueeze(0).to(device).to(dtype=weight_dtype) # torch.Size([2, 2, 3])
        src_inv_wapmat = torch.tensor(image_src_inverse_warpmat256).unsqueeze(0).to(device).to(dtype=weight_dtype)  # torch.Size([2, 2, 3])
        drv_same_wapmat = rearrange(image_tar_warpmat256, "b f c r -> (b f) c r") # torch.Size([2, 12, 2, 3])
        drv_same_inv_wapmat = rearrange(image_tar_inverse_warpmat256, "b f c r -> (b f) c r")
        ref_img = drv_same_img
        ref_wapmat = drv_same_wapmat
        ref_inv_wapmat = drv_same_inv_wapmat
        
        temp = D3DFR_inst.forward_uncropped_tensor_FaceDrive_WithWarpmat_Reenact(#_Reenact512
            rearrange(src_img.unsqueeze(1).repeat(1,video_length,1,1,1),  "b f c h w -> (b f) c h w"), 
            rearrange(src_wapmat.unsqueeze(1).repeat(1,video_length,1,1), "b f c r -> (b f) c r").to(torch.float32),
            rearrange(src_inv_wapmat.unsqueeze(1).repeat(1,video_length,1,1), "b f c r -> (b f) c r").to(torch.float32), 
            ref_img, 
            ref_wapmat.to(torch.float32), 
            ref_inv_wapmat.to(torch.float32)
        )
        id25088 = curricularface_inst.deep_features(warp_affine(src_img, src_wapmat.to(torch.float32), dsize=(test_image_size, test_image_size))).flatten(1)

        # split coeff
        resolution = src_img.shape[-1]
        imRef_d3dBlendSrc = F.interpolate(temp['imRef_d3dBlendSrc'], size=(resolution, resolution), mode="bilinear", align_corners=False,) 
        coeffs = temp['coeffBlendSrcRef']
        id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
        exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
        # ruler angles(x,y,z) for rotation of dim 3
        angles = coeffs[:, 224:227]
        # lighting coeff for 3 channel SH function of dim 27
        gamma = coeffs[:, 227:254]
        translation = coeffs[:, 254:257]  # translation coeff of dim 3
        gaze = coeffs[:, 257:]

        # vae latent
        vae_dtype = pipe.vae.parameters().__next__().dtype
        vae_input = ref_img.to(vae_dtype)
        vae_render_input = imRef_d3dBlendSrc.to(vae_dtype)
        render_latents = pipe.vae.encode(vae_render_input).latent_dist.sample().to(dtype=weight_dtype) 
        render_latents = render_latents * pipe.vae.config.scaling_factor
        render_latents = rearrange(render_latents, "(b f) c h w -> b c f h w", f=video_length)

        clip_input_src_tensors = clip_image_processor(images=src_im_pil, return_tensors="pt").pixel_values.view(-1, 3, 224, 224).to(device)

        # clip embeds
        object_embeds257 = net_vision_encoder(clip_input_src_tensors).last_hidden_state             
        object_hidden_states = mim(
            object_embeds=object_embeds257.repeat(1,video_length,1,1).to(dtype=torch.float32), 
            arcface_embeds=rearrange(id25088.unsqueeze(1).repeat(1, video_length, 1),"b f d -> (b f) d").to(dtype=torch.float32),   
            gaze_embeds=gaze.to(dtype=torch.float32) , 
            exp_embeds=exp_coeff.to(dtype=torch.float32) , 
            tex_embeds=tex_coeff.to(dtype=torch.float32) , 
            gamma_embeds=gamma.to(dtype=torch.float32) , 
            angles_embeds=angles.to(dtype=torch.float32) ,
            translation_embeds=translation.to(dtype=torch.float32) 
        ) # torch.Size([24, 258, 768])
        encoder_hidden_states = rearrange(object_hidden_states,"(b f) l d -> b f l d", f=video_length).mean(1).to(dtype=weight_dtype) # 1id + 257 clip = 258
       
        generator = torch.manual_seed(0)
        prev_frame_num = 3
        latents = pipe(
            render_latents = render_latents,
            prompt_embeds = encoder_hidden_states, 
            video_length=video_length, output_type="latent", # torch.Size([12, 3, 512, 512])
            num_inference_steps=25, generator=generator, guidance_scale=5.0, height=resolution, width=resolution,
            prev_latents=latents.clone() if latents is not None else None,
            prev_frame_num = prev_frame_num
        ).images
 
        # overlap_merge_ratio = torch.linspace(1,0,overlap+2).view(overlap+2,1,1,1)[1:1+overlap].to(device)
        # codenoising
        # if latents_pre is not None:
        #     latents[0: overlap] = latents_pre[video_length-overlap:video_length] * overlap_merge_ratio + latents[0: overlap] * (1-overlap_merge_ratio) # latents 重叠 4 outo f 12
        # latents_pre = latents
    
        images_res = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)
        
        # prev3
        for i in range(current_frame_start_index, min(drive_img_length, current_frame_start_index+video_length)):
            ii = i-current_frame_start_index
            if i < prev_frame_num or ii>=prev_frame_num:
                im_rgb_pil = Image.fromarray(convert_batch_to_nprgb(torch.cat([images_src, images_tar[ii:ii+1], images_res[ii:ii+1]]), 4))
                im_rgb_pil.save(os.path.join(save_path, 'cat_{}'.format(drive_img_list[i])), quality=100)
                # print('cat_{}'.format(drive_img_list[i]))
            
        current_frame_start_index = current_frame_start_index+video_length-prev_frame_num
        print(current_frame_start_index)
        
        # # next3
        # for i in range(current_frame_start_index, min(drive_img_length, current_frame_start_index+video_length)):
        #     ii = i-current_frame_start_index
        #     im_rgb_pil = Image.fromarray(convert_batch_to_nprgb(torch.cat([images_src, images_tar[ii:ii+1], images_res[ii:ii+1]]), 4))
        #     im_rgb_pil.save(os.path.join(save_path, 'cat_{}'.format(drive_img_list[i])), quality=100)
        #     print('cat_{}'.format(drive_img_list[i]))
            
        # current_frame_start_index = current_frame_start_index+video_length-prev_frame_num
        # print(current_frame_start_index)
    



if __name__ == '__main__':
    args = parse_args()
    video_length=args.video_length
    cache_dir = args.cache_dir
    image_model_path = args.image_model_path 
    model_path = args.model_path
    unet2d_path = os.path.join(image_model_path, 'pretrained_unet')
    motion_module_path =  os.path.join(model_path, '3dunet.pth')

    test_image_size = 512
    weight_dtype = torch.float32
    base_model_path = "runwayml/stable-diffusion-v1-5"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pil2tensor = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(mean=0.5, std=0.5)])

    D3DFR_inst = D3DFR_wrapper()
    D3DFR_inst.requires_grad_(False)
    D3DFR_inst.to(device=device)
    D3DFR_inst = D3DFR_inst.eval()
    curricularface_inst = CurricularFaceOurCrop()
    curricularface_inst.requires_grad_(False)
    curricularface_inst.to(device=device)
    curricularface_inst = curricularface_inst.eval()

    unet = UNet3DConditionModel.from_pretrained_2d(unet2d_path, unet_additional_kwargs=unet_additional_kwargs_v3)

    motion_module_state_dict = {}
    temp_dict = torch.load(motion_module_path)
    for k, v in temp_dict.items():
        if "motion_modules." in k or "conv_in." in k:
            motion_module_state_dict[k] = v
    m, u = unet.load_state_dict(motion_module_state_dict, strict=False)
    m = [x for x in m if "motion_modules." in x or "conv_in." in k]
    # print(m)
    unet.to(device, dtype=weight_dtype)
    clip_image_processor = CLIPImageProcessor()
    net_vision_encoder = CLIPVisionModel.from_pretrained(os.path.join(model_path, 'vision_encoder')).to(device, dtype=weight_dtype)
    net_vision_encoder.vision_model.post_layernorm.requires_grad_(False)
    mim = MIMAFacePostfuseModule(768)
    mim.load_state_dict(torch.load(os.path.join(model_path, 'mim.pth')))
    mim = mim.to(device, dtype=torch.float32)

    pipe = MIMAfacePipeline.from_pretrained(
        base_model_path, unet=unet, torch_dtype=weight_dtype, cache_dir=cache_dir, local_files_only=args.local_files_only, requires_safety_checker=False
    ).to(device)

    # speed up diffusion process with faster scheduler and memory optimization
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # # remove following line if xformers is not installed or when using Torch 2.0.
    # pipe.enable_xformers_memory_efficient_attention()
    # # memory optimization.
    # pipe.enable_model_cpu_offload()

    vae_ft_mse = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=cache_dir, torch_dtype=weight_dtype, local_files_only=True).to(device)
    pipe.vae = vae_ft_mse

    src_img_path = args.source_path
    if os.path.isdir(args.target_path):
        drive_img_folder_path = args.target_path
    elif os.path.isfile(args.target_path):
        head, tail = os.path.split(args.target_path)
        ext = tail.split('.')[-1]
        if ext == 'mp4' or ext == 'avi' or ext == 'mov':
            drive_img_folder_path = os.path.join(head, tail.split('.')[0])
            os.makedirs(drive_img_folder_path, exist_ok=True)
            unframe_mp42pil(video_file=args.target_path, k=2, image_folder=drive_img_folder_path) 
        else:
            print('Please specify correct video target path')
            exit()
    else:
        print('Please specify correct target path: directory with images or video (.mp4)')
        exit()

    save_path=os.path.join(args.output_dir, f"{src_img_path.split('/')[-1].split('.')[0]}_{drive_img_folder_path.split('/')[-1].split('.')[0]}")
    os.makedirs(save_path, exist_ok=True)
    infer_one_video(src_img_path, drive_img_folder_path, video_length, save_path)
    save_videos_img2mp4(save_path, os.path.join(save_path, 'video.mp4'))

