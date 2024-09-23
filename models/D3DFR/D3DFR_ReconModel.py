import os
import cv2
import sys
import torch
from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F 
from kornia.geometry import warp_affine
from torchvision.transforms import transforms

from .model_resnet import ResNet50_nofc
from .BFM import BFM09Model


to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def im2tensor(x):
    """
    Args:
        x: np.uint8, [0,255], [H,W,3]

    Returns:
        y: tensor [-1,1], [1,3,H,W]
    """
    return to_tensor(x).unsqueeze(0)

def tensor2im(batch_tensor, n_col=1):
    """
    Args:
        batch_tensor: tensor [B,3,H,W], range=[-1,1], RGB
        n_col:

    Returns:
        np.array [B*H/n_col, W*n_col, 3], range=[0,255], RGB
    """
    img = batch_tensor.detach().cpu().numpy()
    img = ((img+1)*127.5).clip(0,255).astype(np.uint8)
    B, C, H, W = img.shape
    h_multi = int(B/n_col)
    w_multi = n_col
    ## [B,C,H,W]-> [h_multi,w_multi,C,H,W] -> [h_multi,H,w_multi,W,C] -> [ h_multi*H, w_multi*W, C]
    img_plane = img.reshape((h_multi, w_multi, C, H, W)).transpose((0, 3, 1, 4, 2)).reshape((h_multi * H, w_multi * W, C))
    ## if C is 1, 
    if C is 1:
        img_plane = np.tile(img_plane, reps=[1,1,3])
    return img_plane


class D3DFR_wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        ######## load model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ### initial model
        self.model_FR3d = ResNet50_nofc([256, 256], 257 + 4, use_last_fc=False, init_path=None)
        netFR3d_checkpoint = torch.load(f'{self.FILE_DIR}/netG_epoch_25.pth', map_location=lambda storage, loc: storage)
        checkpoint_no_module = self.model_FR3d.state_dict()
        for k, v in netFR3d_checkpoint.items():
            if k.startswith('module'):
                k = k[7:]
            checkpoint_no_module[k] = v
        info = self.model_FR3d.load_state_dict(checkpoint_no_module)
        print(f'load D3DFR ReconNet: {info}')
        self.model_FR3d.eval()
        self.model_FR3d.to(self.device)
        for param in self.model_FR3d.parameters():
            param.requires_grad = False

        #### coeff, mesh reconstruct, render
        model_path = f'{self.FILE_DIR}/BFM09_model_info.mat'
        model_dict = loadmat(model_path)
        self.recon_model = BFM09Model.BFM09ReconModel(model_dict, device='cuda' if torch.cuda.is_available() else 'cpu', img_size=256, focal=1015 * 256 / 224)
        self.recon_model.to(self.device)


    def forward_uncropped_tensor_FaceDrive_WithWarpmat_Reenact(self, imSrc, WarpmatSrc, InverseWarpmatSrc, imRef, WarpmatRef, InverseWarpmatRef):
        """
        :param im_tensor: range [-1,1], B*C*H*W, not cropped
        :return:
        """
        ori_size = imSrc.size(3)
        batch = imSrc.size(0)

        ### 3D 重建和渲染 for Src
        imSrcCrop = warp_affine(imSrc, WarpmatSrc, dsize=(256,256))  
        coeffSrc = self.model_FR3d(imSrcCrop) ## input range[-1,1], return [B 257+4]
        coeffSrc_eye = coeffSrc[:, 257:]  ## shape [B 4]
        idSrc, expSrc, textureSrc, angleSrc, gammaSrc, translationSrc = self.recon_model.split_coeffs(coeffSrc[:, :257])
        
        pred_dict = self.recon_model(coeffSrc[:, :257], render=True)
        ##### warp back to original image
        rendered_imgs = pred_dict['rendered_img']  # [B,256,256,4],  range[0,255]
        out_img_256 = (rendered_imgs[:, :, :, :3] / 255.0).permute(0, 3, 1, 2) * 2 - 1  ## range [-1,1]
        out_mask_256 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)  ## range[0,1], B*1*224*224
        out_img_512 = warp_affine(out_img_256, InverseWarpmatSrc, dsize=(ori_size, ori_size))  ## range[-1,1]
        out_mask_512 = warp_affine(out_mask_256, InverseWarpmatSrc, dsize=(ori_size, ori_size)) ## range[0,1],
        imSrc_Origind3d = (-1 * (1 - out_mask_512) + out_img_512 * out_mask_512)  ### 背景设为-1， 黑色
        mask_src = out_mask_512

        ### 3D 重建和渲染 for Ref
        imRefCrop = warp_affine(imRef, WarpmatRef, dsize=(256, 256))  
        coeffRef = self.model_FR3d(imRefCrop)  ## input range[-1,1], return [B 257+4]
        coeffRef_eye = coeffRef[:, 257:]  ## shape [B 4]
        idRef, expRef, textureRef, angleRef, gammaRef, translationRef = self.recon_model.split_coeffs(coeffRef[:, :257])
        pred_dict = self.recon_model(coeffRef[:, :257], render=True)
        ##### warp back to original image
        rendered_imgs = pred_dict['rendered_img']  # [B,256,256,4],  range[0,255]
        out_img_256 = (rendered_imgs[:, :, :, :3] / 255.0).permute(0, 3, 1, 2) * 2 - 1  ## range [-1,1]
        out_mask_256 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)  ## range[0,1], B*1*224*224
        out_img_512 = warp_affine(out_img_256, InverseWarpmatRef, dsize=(ori_size, ori_size))  ## range[-1,1]
        out_mask_512 = warp_affine(out_mask_256, InverseWarpmatRef, dsize=(ori_size, ori_size))  ## range[0,1],
        imRef_Origind3d = (-1 * (1 - out_mask_512) + out_img_512 * out_mask_512)  ### 背景设为-1， 黑色
        mask_tgt = out_mask_512

        ### 3D 重建和渲染 Blend SrcId RefPosePostion
        coeffBlendSrcRef = torch.cat([idSrc, expRef, textureSrc, angleRef, gammaSrc, translationRef, coeffRef_eye], dim=1)
        pred_dict = self.recon_model(coeffBlendSrcRef[:, :257], render=True)
        rendered_imgs = pred_dict['rendered_img']  # [B,256,256,4],  range[0,255]
        out_img_256 = (rendered_imgs[:, :, :, :3] / 255.0).permute(0, 3, 1, 2) * 2 - 1  ## range [-1,1]
        out_mask_256 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)  ## range[0,1], B*1*224*224
        out_img_512 = warp_affine(out_img_256, InverseWarpmatRef, dsize=(ori_size, ori_size))  ## range[-1,1]
        out_mask_512 = warp_affine(out_mask_256, InverseWarpmatRef, dsize=(ori_size, ori_size))  ## range[0,1],
        imRef_d3dBlendSrc = (-1 * (1 - out_mask_512) + out_img_512 * out_mask_512)  ### 背景设为-1， 黑色
        # VertexPosition = pred_dict['vs']  # [B, 35709, 3], vertex position after rigid transform
        imRef_d3dBlendSrc256 = (-1 * (1 - out_mask_256) + out_img_256 * out_mask_256)
        return {'imSrc_Origind3d': imSrc_Origind3d,
                'imRef_Origind3d':imRef_Origind3d,
                'imRef_d3dBlendSrc':imRef_d3dBlendSrc,
                'imRef_d3dBlendSrc256':imRef_d3dBlendSrc256,
                'coeffBlendSrcRef':coeffBlendSrcRef, ## [261]
                'coeffSrc': coeffSrc,
                'coeffTrg': coeffRef,
                'maskSrc': mask_src,
                'maskTrg': mask_tgt, #out_mask_256
                'lmk': pred_dict['lms_proj']
              }
        
    def get_coeff(self, imSrc, WarpmatSrc):
        imSrcCrop = warp_affine(imSrc, WarpmatSrc, dsize=(256, 256))  ### YT_D3DFR 用的crop256方式
        coeffSrc = self.model_FR3d(imSrcCrop)  ## input range[-1,1], return [B 257+4]
        return coeffSrc[:, :]

    def get_lms(self, coeff):
        pred_dict = self.recon_model(coeff.float(), render=False)
        return pred_dict['lms_proj']


