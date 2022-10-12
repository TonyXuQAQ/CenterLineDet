CUDA_VISIBLE_DEVICES=4 python get_frame_segmentation.py --instance_seg --direction_pred --version v1.0-trainval --savedir HDMapNet/fused_segmentation\
    --dataroot ../dataset/ --model HDMapNet_fusion --bsz 1 --seg_checkpoint_dir ../segmentation_baselines/HDMapNet_for_CenterLineDet/checkpoints/HDMapNet_fusion_best.pt


CUDA_VISIBLE_DEVICES=5 python sampler.py --instance_seg --direction_pred --version v1.0-trainval --savedir HDMapNet\
    --dataroot ../dataset/ --model HDMapNet_fusion --bsz 1 --ROI_half_length 100  --noise 4  --step_size 20 --sample_rate 1
