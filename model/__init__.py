from .hdmapnet import HDMapNet, FusionNet
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar
from .cross_view_encoder import CrossViewTransformer
from .depth3d import Depth3DNet

def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    elif method.lower() == 'IPMNet'.lower():
        model = IPMNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    elif method.lower() == 'CrossViewTransformer'.lower():
        model = CrossViewTransformer()
    elif method.lower() == 'depth3dnet'.lower():
        model = Depth3DNet(data_conf)
    elif method.lower() == 'fusionnet'.lower():
        model = FusionNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, cam=True)
    elif method.lower() == 'fusionnet_cam'.lower():
        model = FusionNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, cam=True)
    elif method.lower() == 'fusionnet_lidar'.lower():
        model = FusionNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, cam=False)
    else:
        raise NotImplementedError

    return model
