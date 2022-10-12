from turtle import forward
import torch
from torch import nn

from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode
from data.utils import gen_dx_bx


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


class HDMapNet(nn.Module):
    def __init__(self, data_conf ,instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(HDMapNet, self).__init__()
        self.camC = 64
        self.downsample = 16

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampler = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, **kwargs):
        x = self.get_cam_feats(img)
        x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(topdown), topdown
    
    def forward_train(self, *args, **kwargs):
        semantic_output = self(*args, **kwargs)
        output = dict(output=semantic_output)
        return output

class IPMNet(HDMapNet):
    def _build_model(self, *args, **kwargs):
        super(IPMNet, self)._build_model(*args, **kwargs)
        x_bound = self.ipm.xbound
        y_bound = self.ipm.xbound
        self.ipm = IPM(x_bound, y_bound, N=6, C=self.camC, extrinsic=False)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans, scale=1):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ks[:, :, :3, :3] = intrins.contiguous()
        Ks[:, :, :2]   = Ks[:, :, :2] / scale
        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        post_RTs[:, :, :3, :3] = post_rots
        post_RTs[:, :, :3, 3] = post_trans


        return Ks, RTs, post_RTs
    
    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, **kwargs):
        _, _, _, H, _ = img.shape
        x = self.get_cam_feats(img)
        _, _, _, h, _ = x.shape
        scale = H // h
        #x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans, scale=scale)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(topdown), topdown

class PathSelector(nn.Sequential):
    def forward(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = super(PathSelector, self).forward(x)
        x = x.view(B, N, 1, imH, imW)
        return x

class FusionNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FusionNet, self).__init__()
        self._build_model(*args, **kwargs)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans, scale=1):
        B, N, _, _ = rots.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ks[:, :, :3, :3] = intrins.contiguous()
        Ks[:, :, :2]   = Ks[:, :, :2] / scale
        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        post_RTs[:, :, :3, :3] = post_rots
        post_RTs[:, :, :3, 3] = post_trans


        return Ks, RTs, post_RTs
    
    def _build_model(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False, cam=True):
        self.camC = 64
        self.downsample = 16

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)
        self.seperate_conv = PathSelector(
            nn.Conv2d(self.camC, self.camC, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.camC),
            nn.Conv2d(self.camC, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.neural_ipm = IPM(ipm_xbound, ipm_ybound,  N=6, C=self.camC, extrinsic=True)
        self.geometric_ipm = IPM(ipm_xbound, ipm_ybound,  N=6, C=self.camC, extrinsic=False)

        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampler = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)

        self.lidar = lidar
        self.cam = cam
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        if lidar and cam:
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        elif lidar:
            self.bevencode = BevEncode(inC=128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, **kwargs):
        x = self.get_cam_feats(img)
        
        path_selector = self.seperate_conv(x) # [B, 1, H, W]

        # neural transform
        x_neural = x.clone()
        x_neural = self.view_fusion(x_neural)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(torch.eye(3).cuda(), rots, trans, post_rots, post_trans, scale=self.downsample)
        neural_topdown = self.neural_ipm(x_neural, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)

        # Geometric Transform
        x_geo = x.clone()
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans, scale=self.downsample)
        geo_topdown = self.geometric_ipm(x_geo, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)

        topdown = neural_topdown + geo_topdown
        topdown = self.up_sampler(topdown)
        if self.lidar and self.cam:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        elif self.lidar:
            topdown = self.pp(lidar_data, lidar_mask)
        elif self.cam:
            pass
        else:
            raise Exception("Wrong model choice!")
        return self.bevencode(topdown), topdown
    
    def forward_train(self, *args, **kwargs):
        semantic_output = self(*args, **kwargs)
        output = dict(output=semantic_output)
        return output
