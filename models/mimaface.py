import cv2
import types
import copy
import random
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from kornia.geometry.transform import warp_affine
from typing import Any, Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, StableDiffusionPipeline
from transformers import AutoImageProcessor, Dinov2Model, Dinov2PreTrainedModel, CLIPImageProcessor, CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextTransformer, CLIPPreTrainedModel, CLIPModel
from .animatediff.unet import UNet3DConditionModel, unet_additional_kwargs_v3
from .animatediff.resnet import InflatedConv3d
from .transformer import Transformer


transform = transforms.Compose([ 
                transforms.ToTensor(), 
                transforms.Normalize(mean=0.5, std=0.5)])


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x




class CLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
    ):
        # import pdb;pdb.set_trace()
        model = CLIPModel.from_pretrained("/youtu_xuanyuan_shuzhiren_2906355_cq10/private/hyy/face-adapter-inference/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff/")
        miss_keys, unexpected_keys = model.load_state_dict(torch.load(global_model_name_or_path), strict=False)
        print(unexpected_keys)
        # model = CLIPModel.from_pretrained(global_model_name_or_path)
        
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return CLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size


    def forward(self, object_pixel_values):
        # import pdb;pdb.set_trace()
        b, num_objects, c, h, w = object_pixel_values.shape

        object_pixel_values = object_pixel_values.view(b * num_objects, c, h, w)

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        object_pixel_values = self.vision_processor(object_pixel_values)
        output = self.vision_model(object_pixel_values)
        object_embeds = output[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, num_objects, 1, -1) # torch.Size([1, 1, 1, 768])
        object_embeds257 = output[0]
        object_embeds257 = object_embeds257.view(b, num_objects, 257, -1) # torch.Size([1, 1, 257, 1024])
        return object_embeds, object_embeds257


class MIMAFacePostfuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # self.transform_dim_clip = nn.Linear(768, 768)
        self.transform_dim_clip = nn.Linear(1024, 768)
        # self.transform_dim_arcface = nn.Linear(512, 768)
        self.transform_dim_arcface = nn.Linear(25088, 768)
        self.transform_dim_gaze = nn.Linear(4, 768)
        self.transform_dim_exp = nn.Linear(64, 768)
        self.transform_dim_tex = nn.Linear(80, 768)
        self.transform_dim_gamma = nn.Linear(27, 768)
        self.transform_dim_angles = nn.Linear(3, 768)
        self.transform_dim_translation = nn.Linear(3, 768)
        self.mapper = Transformer(1,768,3,1) 
        # self.mapper = MLP(768,768,768)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=768//64)
        # self.mapper  = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.layer_norm = nn.LayerNorm(768)

    def forward(
        self,
        object_embeds,
        text_embeds=None,
        arcface_embeds=None,
        gaze_embeds=None,
        exp_embeds=None,
        tex_embeds=None,
        gamma_embeds=None,
        angles_embeds=None,
        translation_embeds=None,
    ) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        output = self.transform_dim_clip(object_embeds.squeeze(0))
        arcface_output = self.transform_dim_arcface(arcface_embeds.unsqueeze(1))
        gaze_output = self.transform_dim_gaze(gaze_embeds.unsqueeze(1))
        exp_output = self.transform_dim_exp(exp_embeds.unsqueeze(1))
        tex_output = self.transform_dim_tex(tex_embeds.unsqueeze(1))
        gamma_output = self.transform_dim_gamma(gamma_embeds.unsqueeze(1))
        angles_output = self.transform_dim_angles(angles_embeds.unsqueeze(1))
        translation_output = self.transform_dim_translation(translation_embeds.unsqueeze(1))
        output = torch.cat([arcface_output, gaze_output, exp_output, tex_output, gamma_output, angles_output, translation_output, output], dim=1)
        # output = torch.cat([ arcface_output, output], dim=1)
        output = self.mapper(output)
        output = self.layer_norm(output)
        return output
    
# class MIMAFacePostfuseModule(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.transform_dim_clip = nn.Linear(1024, 768)
#         self.transform_dim_arcface = nn.Linear(25088, 768)
#         self.transform_dim_coeff = nn.Linear(181, 1024)
#         self.mapper = Transformer(1,768,3,1) 
#         self.layer_norm = nn.LayerNorm(768)
#         self.cross_attention = nn.MultiheadAttention(1024, 8)
#         self.layer_norm0 = nn.LayerNorm(768)
        

#         # self.query = nn.Parameter(torch.randn((1, 258, 768)))
#         # decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=768//64, batch_first=True)
#         # self.mim = nn.TransformerDecoder(decoder_layer, num_layers=3)

#         # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=768//64)
#         # self.mapper  = nn.TransformerEncoder(encoder_layer, num_layers=3)

#     def forward(
#         self,
#         object_embeds,
#         text_embeds=None,
#         arcface_embeds=None,
#         gaze_embeds=None,
#         exp_embeds=None,
#         tex_embeds=None,
#         gamma_embeds=None,
#         angles_embeds=None,
#         translation_embeds=None,
#     ) -> torch.Tensor:
        
#         appearance_features = rearrange(object_embeds, "b f l d -> (b f) l d") # torch.Size([2, 257, 1024])
#         bs = appearance_features.shape[0]
#         motion_features = torch.cat([
#             tex_embeds, 
#             gaze_embeds,  # torch.Size([24, 4])
#             gamma_embeds, 
#             exp_embeds, 
#             angles_embeds,
#             translation_embeds # torch.Size([24, 3])
#             ], dim=-1).unsqueeze(1) # torch.Size([24, 1, 181])
        
#         appearance_features = appearance_features.transpose(0, 1) # torch.Size([2, 257, 1024])
#         motion_features = motion_features.transpose(0, 1) # torch.Size([1, 24, 181])
#         motion_features = self.transform_dim_coeff(motion_features) # torch.Size([1, 24, 1024])
#         modulated_features, _ = self.cross_attention(appearance_features, motion_features, motion_features)
#         modulated_features = modulated_features + appearance_features # torch.Size([257, 24, 1024])
#         modulated_features = modulated_features.transpose(0, 1) # torch.Size([24, 257, 1024])
#         modulated_features = self.transform_dim_clip(modulated_features)
#         modulated_features = self.layer_norm0(modulated_features)
#         arcface_output = self.transform_dim_arcface(arcface_embeds.unsqueeze(1))

#         output = torch.cat([arcface_output, modulated_features], dim=1) # torch.Size([1, 1, 768])

#         output = self.mapper(output)
#         output = self.layer_norm(output)
#         return output




# class MIM(nn.Module):

#     def __init__(self, id_dim=512, text_hidden_size=1024, max_length=77, num_layers=0):
#         super(ID2Token, self).__init__()
        
        
#         self.text_hidden_size = text_hidden_size
#         self.id_dim = id_dim
#         self.id_proj = nn.Linear(id_dim, text_hidden_size)
        
#         if num_layers>0:
#             self.query = nn.Parameter(torch.randn((1, max_length, text_hidden_size)))
#             decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
#             self.id2t = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            
#         else:
#             self.id2t = None
        
#         # self.fc_out = nn.Linear(id_dim, self.text_hidden_size)
#         # self.layernorm = nn.LayerNorm(text_hidden_size)

#     def forward(self, x):
#         b=x.shape[0]
#         out = self.id_proj(x).view(b,-1,self.text_hidden_size)
#         if self.id2t is not None:
#             out = self.id2t(self.query.repeat(b,1,1), out)
        
#         # out = self.fc_out(out)
#         # out = self.layernorm(out)
#         return out
    


class MIMAFaceModel(nn.Module):
    def __init__(self, image_encoder, vae, unet, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.revision = args.revision
        self.non_ema_revision = args.non_ema_revision
        self.video_length = args.video_length
        self.postfuse_module = MIMAFacePostfuseModule(768)
        self.unet_in = args.unet_in
        self.train_resolution = args.train_resolution

    @staticmethod
    def from_pretrained(args, tokenizer):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", 
            revision=args.revision
        )
        
        if args.pretrained_unet_path is not None:
            unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_unet_path, unet_additional_kwargs=unet_additional_kwargs_v3)
        else:
            unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", unet_additional_kwargs=unet_additional_kwargs_v3)
        
        # modify conv_in channel
        old_weights = unet.conv_in.weight
        old_bias = unet.conv_in.bias
        new_conv1 = InflatedConv3d(
            13, old_weights.shape[0],
            kernel_size=unet.conv_in.kernel_size,
            stride=unet.conv_in.stride,
            padding=unet.conv_in.padding,
            bias=True if old_bias is not None else False)
        param = torch.zeros((320,9,3,3),requires_grad=True)
        new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights,param),dim=1))
        if old_bias is not None:
            new_conv1.bias = old_bias
        unet.conv_in = new_conv1
        unet.config["in_channels"] = 13   
            
        if args.pretrained_motion_module_path is not None:
            motion_module_state_dict = {}
            temp_dict = torch.load(args.pretrained_motion_module_path)
            for k, v in temp_dict.items():
                if "motion_modules." in k:
                    motion_module_state_dict[k] = v
            miss_keys, unexpected_keys = unet.load_state_dict(motion_module_state_dict, strict=False)
            # print(args.pretrained_motion_module_path, unexpected_keys)

        image_encoder = CLIPImageEncoder.from_pretrained(
            args.image_encoder_name_or_path,
        )

        return MIMAFaceModel(image_encoder, vae, unet, tokenizer, args)

    def set_input_video(self, batch):
        self.src_img = images_src = batch['image_src'] # torch.Size([2, 3, 512, 512])
        self.src_img_clip_input_tensors = batch['image_src_clip'] # torch.Size([2, 3, 224, 224])
        self.src_img_crop256 = batch['image_src_crop256'] # torch.Size([2, 3, 256, 256])
        self.drv_same_img = rearrange(batch['image_tar'], "b f c h w -> (b f) c h w")
        self.drv_same_img_crop256 = rearrange(batch['image_tar_crop256'], "b f c h w -> (b f) c h w")
        
        self.src_wapmat = batch['image_src_warpmat256'] # torch.Size([2, 2, 3])
        self.src_inv_wapmat = batch['image_src_inverse_warpmat256'] # torch.Size([2, 2, 3])
        self.drv_same_wapmat = rearrange(batch['image_tar_warpmat256'], "b f c r -> (b f) c r") # torch.Size([2, 12, 2, 3])
        self.drv_same_inv_wapmat = rearrange(batch['image_tar_inverse_warpmat256'], "b f c r -> (b f) c r")

        self.ref_img = self.drv_same_img
        self.ref_img_crop256 = self.drv_same_img_crop256 
        self.ref_wapmat = self.drv_same_wapmat
        self.ref_inv_wapmat = self.drv_same_inv_wapmat

    def split_coeffs(self, coeffs):
        with torch.no_grad():
            id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
            exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
            tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
            # ruler angles(x,y,z) for rotation of dim 3
            angles = coeffs[:, 224:227]
            # lighting coeff for 3 channel SH function of dim 27
            gamma = coeffs[:, 227:254]
            translation = coeffs[:, 254:257]  # translation coeff of dim 3
            gaze = coeffs[:, 257:]
        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, gaze

    def forward_coeff(self, temp, id25088):
        with torch.no_grad():
            self.imRef_d3dBlendSrc = temp['imRef_d3dBlendSrc']
            resolution = self.src_img.shape[-1]
            self.imRef_d3dBlendSrc = F.interpolate(temp['imRef_d3dBlendSrc'], size=(resolution, resolution), mode="bilinear", align_corners=False,) 
            self.id_coeff, self.exp_coeff, self.tex_coeff, self.angles, self.gamma, self.translation, self.trg_gaze = self.split_coeffs(temp['coeffBlendSrcRef'])
            self.id25088 = id25088
            

