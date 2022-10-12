CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --lr 1e-3 --instance_seg --direction_pred --savedir FusionNet \
    --model fusionnet --bsz 12 --nepochs 50 --train_set_selection all