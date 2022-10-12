CUDA_VISIBLE_DEVICES=2 python inference.py --instance_seg --direction_pred \
--savedir HDMapNet --model HDMapNet_fusion --bsz 1 \
--seg_checkpoint_dir ./HDMapNet/checkpoints/HDMapNet_fusion_best.pt 