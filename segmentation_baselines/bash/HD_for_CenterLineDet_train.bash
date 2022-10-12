CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 train.py\
 --lr 1e-3 --instance_seg --direction_pred --savedir HDMapNet_for_CenterLineDet\
 --model HDMapNet_fusion --bsz 12 --nepochs 50 --train_set_selection train1