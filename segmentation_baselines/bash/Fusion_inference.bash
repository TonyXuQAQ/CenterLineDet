CUDA_VISIBLE_DEVICES=3 python inference.py --instance_seg --direction_pred \
 --savedir FusionNet --model fusionnet --bsz 1 \
 --seg_checkpoint_dir ./FusionNet/checkpoints/fusionnet_best.pt 



