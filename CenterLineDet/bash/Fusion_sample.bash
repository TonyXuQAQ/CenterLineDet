CUDA_VISIBLE_DEVICES=4 python get_frame_segmentation.py --instance_seg --direction_pred --version v1.0-trainval --savedir FusionNet/fused_segmentation\
    --dataroot ../dataset/ --model fusionnet --bsz 1 --seg_checkpoint_dir ../segmentation_baselines/FusionNet_for_CenterLineDet/checkpoints/fusionnet_best.pt
                                                                    

CUDA_VISIBLE_DEVICES=5 python sampler.py --instance_seg --direction_pred --version v1.0-trainval --savedir FusionNet\
    --dataroot ../dataset/ --model fusionnet --bsz 1 --ROI_half_length 100  --noise 4 --step_size 20 --sample_rate 1
    
