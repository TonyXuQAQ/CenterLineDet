CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 train.py\
 --lr 1e-3 --instance_seg --direction_pred --savedir FusionNet_for_CenterLineDet\
 --model fusionnet --bsz 12 --nepochs 50 --train_set_selection train1