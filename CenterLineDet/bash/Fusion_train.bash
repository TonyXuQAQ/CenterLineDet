CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 train.py\
   --instance_seg --direction_pred --version v1.0-trainval --savedir FusionNet --backbone resnet101 \
    --dataroot ../dataset/ --model fusionnet --bsz 128 --nworkers 8 --ROI_half_length 100 --eos_coef 0.25\
    --training_mode BC --nepochs 100 --lr 1e-4 \
    --binary_thr 0.05 --area_thr 8 --logit_thr 0.9 --filter_distance 10 --world_alignment_distance 1\
   