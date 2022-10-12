CUDA_VISIBLE_DEVICES=4 python get_frame_segmentation.py --instance_seg --direction_pred --version v1.0-trainval --savedir FusionNet/fused_segmentation_all\
    --dataroot ../dataset/ --model fusionnet --bsz 1 --seg_checkpoint_dir ../segmentation_baselines/FusionNet/checkpoints/fusionnet_best.pt
