
import os
import sys
import cv2
import random
import imageio
import numpy as np
import argparse
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
from pipeline_image import MIMAfacePipeline
from util import save_videos_img2mp4, convert_batch_to_nprgb
from models.mimaface import MIMAFacePostfuseModule
from models.animatediff.unet import UNet3DConditionModel
from models.D3DFR.D3DFR_ReconModel import D3DFR_wrapper
from models.ArcFace.CurricularFace_wrapper import CurricularFaceOurCrop


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--source_path', type=str, default='./examples/source/bengio.jpg', help="path to source identity")
    parser.add_argument('--target_path', type=str, default='./examples/target/0000025.jpg', help="path to target pose")
    parser.add_argument('--output_dir', type=str, default='./examples/result', help="path to save the results")
    parser.add_argument('--image_model_path', type=str, default='./checkpoints/image', help="path to save the results")
    parser.add_argument(
        "--video_length", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./hub',
        help="The directory where the downloaded models and datasets will be stored.",
    )    
    parser.add_argument(
        "--local_files_only",
        action="store_true",
    )
    args = parser.parse_args()
    return args

args = parse_args()
video_length=args.video_length
cache_dir = args.cache_dir
image_model_path = args.image_model_path 

torch.set_grad_enabled(False)
weight_dtype = torch.float32
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
test_image_size = 512
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


base_model_path = "runwayml/stable-diffusion-v1-5"
pipe = MIMAfacePipeline.from_pretrained(
    base_model_path, torch_dtype=weight_dtype, cache_dir=cache_dir, local_files_only=args.local_files_only, requires_safety_checker=False
).to(device)
pipe.safety_checker=None
# pretrained unet
pretrained_unet_path = os.path.join(image_model_path, 'pretrained_unet')
pipe.unet = UNet2DConditionModel.from_pretrained(pretrained_unet_path, torch_dtype=weight_dtype, local_files_only=args.local_files_only).to(device)
    
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# # remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# # memory optimization.
# pipe.enable_model_cpu_offload()

vae_ft_mse = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=cache_dir, torch_dtype=weight_dtype, local_files_only=args.local_files_only).to(device)
pipe.vae = vae_ft_mse

clip_image_processor = CLIPImageProcessor()

# "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
net_vision_encoder = CLIPVisionModel.from_pretrained(os.path.join(image_model_path, 'vision_encoder')).to(device)
net_vision_encoder.vision_model.post_layernorm.requires_grad_(False)
mim = MIMAFacePostfuseModule(768)
mim.load_state_dict(torch.load(os.path.join(image_model_path, 'mim.pth')))
mim = mim.to(device)



def infer_one_video(src_img_path, drive_img_path, video_length, save_path, overlap=3, clip=0):
    
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
    # ======
    
    drive_im_pil = Image.open(drive_img_path).convert("RGB")
    crop_drive=False
    boxes, _, landmarks = mtcnn.detect(drive_im_pil, landmarks=True)
    if crop_drive:
        crop_ratio = -1
        if crop_ratio>0:
            dets = boxes[0]
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
    image_tar_inverse_warpmat256=cv2.invertAffineTransform(image_tar_warpmat256)
    
    drive_im_crop256 = cv2.warpAffine(np.array(drive_im_pil), image_tar_warpmat256, (256, 256), flags=cv2.INTER_LINEAR)
    images_tar = pil2tensor(drive_im_pil).view(1,3,test_image_size,test_image_size).to(device) 
    # ======

    image_tar_warpmat256 = torch.tensor(image_tar_warpmat256).unsqueeze(0).unsqueeze(0).to(device).to(dtype=weight_dtype) 
    image_tar_inverse_warpmat256 = torch.tensor(image_tar_inverse_warpmat256).unsqueeze(0).unsqueeze(0).to(device).to(dtype=weight_dtype) ###### (10, 2, 3)
    
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
    
    # import pdb;pdb.set_trace()
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

    clip_input_src_tensors = clip_image_processor(images=src_im_pil, return_tensors="pt").pixel_values.view(-1, 3, 224, 224).to(device)

    # clip embeds
    object_embeds257 = net_vision_encoder(clip_input_src_tensors).last_hidden_state
    object_hidden_states = mim(
        object_embeds=object_embeds257.repeat(1,video_length,1,1).to(dtype=weight_dtype), 
        arcface_embeds=rearrange(id25088.unsqueeze(1).repeat(1, video_length, 1),"b f d -> (b f) d").to(dtype=weight_dtype),   
        gaze_embeds=gaze.to(dtype=weight_dtype) , 
        exp_embeds=exp_coeff.to(dtype=weight_dtype) , 
        tex_embeds=tex_coeff.to(dtype=weight_dtype) , 
        gamma_embeds=gamma.to(dtype=weight_dtype) , 
        angles_embeds=angles.to(dtype=weight_dtype) ,
        translation_embeds=translation.to(dtype=weight_dtype) 
    ) 
    encoder_hidden_states = object_hidden_states
    generator = torch.manual_seed(0)
    latents = pipe(
        render_latents=render_latents, # torch.Size([1, 4, 64, 64])
        prompt_embeds=encoder_hidden_states,  # torch.Size([1, 258, 768])
        output_type="latent", # torch.Size([12, 3, 512, 512])
        num_inference_steps=25, generator=generator, guidance_scale=5.0, height=resolution, width=resolution,
    ).images
    
    images_res = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)
    im_rgb_pil = Image.fromarray(convert_batch_to_nprgb(torch.cat([images_src, images_tar, images_res, imRef_d3dBlendSrc.view(1,3,test_image_size,test_image_size)]), 4))
    im_rgb_pil.save(save_path, quality=100)

src_img_path = args.source_path
drive_img_path = args.target_path
save_path=f"examples/result/{src_img_path.split('/')[-1].split('.')[0]}_{drive_img_path.split('/')[-1]}"
os.makedirs(os.path.split(save_path)[0], exist_ok=True)
infer_one_video(src_img_path, drive_img_path, video_length, save_path)
    
