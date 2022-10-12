import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from model.base import BevEncode
from .hdmapnet import CamEncode
from .pointpillar import PillarBlock, PointNet, points_to_voxels
import torch_scatter

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def depth_to_disp(depth, min_depth, max_depth):
    """Convert network's depth prediction into sigmoid output
    """

    disp = (1 / depth - 1 / max_depth) / (1 / min_depth - 1 / max_depth)
    return disp

def inverse_sigmoid(x):
    """Inverse sigmoid function
    """
    return torch.log(x / (1 - x + 1e-8))

def cam_points2BEV(cam_points):
    """ Transform Campoints [..., 3] [x,y,z] 
    """


class PointPillarEncoderCustomed(nn.Module):
  def __init__(self, feature, C, xbound, ybound, zbound):
    super(PointPillarEncoderCustomed, self).__init__()
    self.xbound = xbound
    self.ybound = ybound
    self.zbound = zbound
    self.pn = PointNet(13 + feature, 64)
    self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
    self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
    self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
    self.up1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
      nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
      nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, C, 1),
    )

  def forward(self, points, points_mask):
    points_xyz = points[:, :, :3]
    points_feature = points[:, :, 3:]
    voxels = points_to_voxels(
      points_xyz, points_mask, self.xbound, self.ybound, self.zbound
    )
    points_feature = torch.cat(
      [points, # 5
       torch.unsqueeze(voxels['voxel_point_count'], dim=-1), # 1
       voxels['local_points_xyz'], # 3
       voxels['point_centroids'], # 3
       points_xyz - voxels['voxel_centers'], # 3
      ], dim=-1
    )
    points_feature = self.pn(points_feature, voxels['points_mask'])
    voxel_feature = torch_scatter.scatter_mean(
      points_feature,
      torch.unsqueeze(voxels['voxel_indices'], dim=1),
      dim=2,
      dim_size=voxels['num_voxels'])
    batch_size = points.size(0)
    voxel_feature = voxel_feature.view(batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
    voxel_feature1 = self.block1(voxel_feature)
    voxel_feature2 = self.block2(voxel_feature1)
    voxel_feature3 = self.block3(voxel_feature2)
    voxel_feature1 = self.up1(voxel_feature1)
    voxel_feature2 = self.up2(voxel_feature2)
    voxel_feature3 = self.up3(voxel_feature3)
    voxel_feature = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
    return self.conv_out(voxel_feature).transpose(3, 2)

class SupervisedMonoDepthLoss(nn.Module):
    def __init__(self, down_sample=8):
        super().__init__()
        self.down_sample = down_sample

    def forward(self, depthmap, lidar, intrinsics, extrinsics):
        B, N, _, H, W = depthmap
        lidar_xyz = lidar[..., 0:3].unsqueeze(1).repeat([1, N, 1, 1, 1])
        _, N_pts, _ = lidar_xyz.shape # [B, N_pts, 3]
        pass

class Depth3DNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Depth3DNet, self).__init__()
        self._build_model(*args, **kwargs)

    def _build_depth_bins(self, min_depth, max_depth, num_bins):
        inv_depth_min = np.log(min_depth)
        inv_depth_max = np.log(max_depth)

        inv_depth_bins = torch.arange(inv_depth_min, inv_depth_max, (inv_depth_max - inv_depth_min) / num_bins)
        depth_bins = torch.exp(inv_depth_bins)
        self.register_buffer("depth_bins", depth_bins)
    
    def _gather_activation(self, x:torch.Tensor)->torch.Tensor:
        """Decode the output of the cost volume into a encoded depth feature map.

        Args:
            x (torch.Tensor): The output of the cost volume of shape [B, num_depth_bins, H, W]

        Returns:
            torch.Tensor: Encoded depth feature map of shape [B, 1, H, W]
        """

        activated = torch.softmax(x, dim=2)
        encoded_depth = torch.sum(activated * self.depth_bins.reshape(1, -1, 1, 1), dim=2, keepdim=True) # type: ignore
        activation = inverse_sigmoid(depth_to_disp(encoded_depth, self.min_depth, self.max_depth))
        return activation



    def _build_model(self, data_conf):
        self.C = 64
        self.downsample = 4
        self.camencode = CamEncode(self.C)

        self.depth_decode = nn.Sequential(
            nn.Conv2d(self.C, self.C, 3, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.C, self.C // 2, 3, padding=1),
            nn.BatchNorm2d(self.C // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.C // 2, 16 + self.C // 4, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

        self.pp = PointPillarEncoderCustomed(self.C // 4, 128, data_conf['xbound'],
        data_conf['ybound'], data_conf['zbound'])
        self.bevencode = BevEncode(inC=128, outC=data_conf['num_channels'], instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=37)

        self.min_depth = 0.1
        self.max_depth = 100
        self._build_depth_bins(self.min_depth, self.max_depth, 16)
    
    def get_Ks_RTs(self, intrins, rots, trans, post_rots):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ks[:, :, 0:3, 0:3] = intrins
        RTs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        RTs[:, :, :3, :3] = rots
        RTs[:, :, :3, 3] = trans

        Ks[:, :, 0:1, :] *= post_rots[:, :, 0:1, 0:1] / self.downsample
        Ks[:, :, 1:2, :] *= post_rots[:, :, 1:2, 1:2] / self.downsample

        return Ks, RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = self.depth_decode(x)
        _, outputC, _, _ = x.shape
        x = x.view(B, N, outputC, imH//self.downsample, imW//self.downsample)
        return x

    def extract_feat(self, x):
        B, N, C_1, H_8, W_8 = x.shape
        depth_act = x[:, :, 0:16, :, :]
        features = x[:, :, 16:, :, :]
        return depth_act, features

    def compute_depth(self, depth_act):
        disp = self.sigmoid(
            self._gather_activation(depth_act)
        )
        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        depth_output = dict(
            disp=disp, depth=depth #[B, N, 1, H_4, W_4]
        )
        return depth_output

    def inference_point_cloud(self, features, depth, intrinsics, extrinsics):
        B, N, C, H_8, W_8 = features.shape

        # features : B, N, C, H, W -> B, N*H*W, C
        features = features.permute([0, 1, 3, 4, 2]).reshape([B, N*H_8*W_8, C])

        depth    = depth.reshape(B*N, 1, H_8, W_8)  # [B*N, 1, H, W]
        K = intrinsics[:, :, 0:3, 0:3].reshape(B*N, 3, 3) # [B*N, 3, 3]
        E = extrinsics.reshape([B*N, 4, 4]) #[B*N, 4, 4]
        import kornia
        # project to camera frame
        depth_3d  = kornia.geometry.depth_to_3d(depth, K) # [B*N, 3, H_8, W_8]
        depth_3d = depth_3d.permute([0, 2, 3, 1]).reshape([B*N, H_8*W_8, 3])

        # transform camera points to bev with extrinsic parameters
        bev_points  = kornia.geometry.transform_points(
            E, depth_3d
        ) #[B*N, H*W, 3]
        bev_points = bev_points.reshape(B, N, H_8*W_8, 3).reshape([B, N*H_8*W_8, 3])

        cated_pts = torch.cat(
            [bev_points, features], dim=-1
        ) #[B, N'', 3+C]
        return cated_pts


    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, **kwargs):
        img_feats = self.get_cam_feats(img) # [B, N, C, H // 8, W // 8]
        depth_act, features = self.extract_feat(img_feats)
        depth_output = self.compute_depth(depth_act)

        intrinsics, extrinsics = self.get_Ks_RTs(intrins, rots, trans, post_rots)

        point_cloud = self.inference_point_cloud(features, 
            depth_output['depth'], intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        B, N_pts, _ = point_cloud.shape
        pc_mask = torch.ones([B, N_pts], device=img.device)
        pc_features = self.pp(point_cloud, pc_mask)
        topdown = self.bevencode(pc_features)
        return topdown

    def depth_loss(self, depth_pred, pc_depth):
        B, N, H, W = pc_depth.shape
        depth_pred = torch.nn.functional.upsample(depth_pred[:, :, 0], [H, W]) #[B, N, H, W]
        mask = pc_depth > 0.5
        
        diff = depth_pred[mask] - pc_depth[mask]
        loss = diff.abs().mean()
        return loss

    def forward_train(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, depth, **kwargs):

        img_feats = self.get_cam_feats(img) # [B, N, C, H // 8, W // 8]
        depth_act, features = self.extract_feat(img_feats)
        depth_output = self.compute_depth(depth_act)

        intrinsics, extrinsics = self.get_Ks_RTs(intrins, rots, trans, post_rots)

        point_cloud = self.inference_point_cloud(features, 
            depth_output['depth'], intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        B, N_pts, _ = point_cloud.shape
        pc_mask = torch.ones([B, N_pts], device=img.device)
        pc_features = self.pp(point_cloud, pc_mask)
        topdown = self.bevencode(pc_features)

        output = dict(output=topdown, loss=self.depth_loss(
            depth_output['depth'], depth
        ))
        return output
