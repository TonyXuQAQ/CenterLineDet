CUDA_VISIBLE_DEVICES=3 python inference.py --instance_seg --direction_pred --version v1.0-trainval \
    --savedir HDMapNet --backbone resnet101 --dataroot ../dataset/ --model HDMapNet_fusion --bsz 1\
    --centerlinedet_checkpoint_dir ./HDMapNet/checkpoints/center_HDMapNet_fusion_best.pt --ROI_half_length 100\
    --binary_thr 0.05 --area_thr 8 --logit_thr 0.9 --filter_distance 10 --world_alignment_distance 1
