
import os
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch
from torch import Tensor
import torch.nn as nn
from kornia.geometry import warp_affine
from .curricularface_model import IR_101


class CurricularFaceOurCrop(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        model = IR_101(input_size=[112, 112])
        checkpoint = torch.load(f'{self.FILE_DIR}/CurricularFace_Backbone.pth', map_location=lambda storage, loc: storage)
        checkpoint_no_module = {}
        for k, v in checkpoint.items():
            if k.startswith('module'):
                k = k[7:]
            checkpoint_no_module[k] = v
        info = model.load_state_dict(checkpoint_no_module)
        print(f"Load curricularface_model: {info}")

        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.net = model.to(self.device)

    def forward(self, im):
        """
        im: [-1,1], Crop256 or Crop512 or Crop1024
        return id_feature
        """
        if im.size(2) != 256:
            im = F.interpolate(im, size=[256, 256], mode='bilinear')
        # import pdb;pdb.set_trace()
        id_feature = self.net(F.interpolate(im[:, :, :224, 16:240], size=[112, 112], mode='bilinear')) ## crop256 --> crop224 --> crop112
        # id_feature = F.normalize(id_feature, dim=-1, p=2)
        return id_feature
    def deep_features(self, im):
        """
        im: [-1,1], Crop256 or Crop512 or Crop1024
        return id_feature
        """
        if im.size(2) != 256:
            im = F.interpolate(im, size=[256, 256], mode='bilinear')

        id_feature = self.net.deep_features(F.interpolate(im[:, :, :224, 16:240], size=[112, 112], mode='bilinear')) ## crop256 --> crop224 --> crop112
        # id_feature = F.normalize(id_feature, dim=-1, p=2)
        return id_feature
