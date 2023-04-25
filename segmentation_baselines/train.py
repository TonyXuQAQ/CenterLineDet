import os
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.distributed as dist
import shutil

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F

import sys 
sys.path.append('..')
from config.get_args import get_args
from model.loss import SimpleLoss
from data.dataset import segmentation_dataset
from model import get_model

from inference import evaluation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

def train(args):
    # ==============
    data_conf = {
        'num_channels': 2,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'is_depth': True
    }
    if args.local_rank==0:
        # ============== 
        args.train_save_dir = f'./{args.savedir}/train'
        args.eval_save_dir = f'./{args.savedir}/val'
        create_directory(f'./{args.savedir}/tensorboard',delete=True)
        create_directory(f'./{args.savedir}/tensorboard_past')
        create_directory(f'./{args.savedir}/val/single/gt_mask',delete=True)
        create_directory(f'./{args.savedir}/val/single/gt_point_mask',delete=True)
        create_directory(f'./{args.savedir}/val/multi/gt_mask',delete=True)
        create_directory(f'./{args.savedir}/val/single/pred_mask',delete=True)
        create_directory(f'./{args.savedir}/val/single/pred_point_mask',delete=True)
        create_directory(f'./{args.savedir}/val/multi/pred_mask',delete=True)
        create_directory(f'./{args.savedir}/train',delete=True)
        create_directory(f'./{args.savedir}/checkpoints')

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(f'cuda:{args.local_rank}')
    train_loader, train_sampler, val_loader = segmentation_dataset(args,data_conf,train_set_selection=args.train_set_selection)
    model = get_model(args.model, data_conf,args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

    opt = torch.optim.AdamW([{'params':model.parameters()},
    ], lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    
    if args.local_rank==0:
        writer = SummaryWriter(args.savedir+'/tensorboard')
    loss_fn = SimpleLoss(args.pos_weight).cuda()
    if args.local_rank==0:
        print(f'============ Start training {args.model}')
    #=====================================
    model.train()
    best_f1 = 0
    for epoch in range(args.nepochs):
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), unit='img') as pbar:
            for i, data in enumerate(train_loader):
                # Segmentation head
                semantic_pred, _ = model(data['imgs'].cuda(), 
                                        data['trans'].cuda(),
                                        data['rots'].cuda(),
                                        data['intrins'].cuda(),
                                        data['post_trans'].cuda(),
                                        data['post_rots'].cuda(),
                                        data['lidar_data'].cuda(),
                                        data['lidar_mask'].cuda(),
                                        data['car_trans'].cuda(),
                                        data['yaw_pitch_roll'].cuda())
                segment_mask = torch.cat([data['segment_mask'].unsqueeze(1).cuda(),data['initial_candidate_mask'].unsqueeze(1).cuda()],dim=1)
                loss_dict = loss_fn(semantic_pred, segment_mask)
                loss_seg = loss_dict['loss_mask']
                opt.zero_grad()
                loss_seg.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                
                if args.local_rank==0:
                    writer.add_scalar('train/loss_seg', loss_seg, i+epoch*len(train_loader))
                    dst = Image.new('RGB',(200*2+5,200*2+5))
                    crop_segment_mask = Image.fromarray((segment_mask[0,0].cpu().detach().numpy()*255).astype(np.uint8))
                    crop_pred_segment_mask = Image.fromarray((semantic_pred[0,0].sigmoid().cpu().detach().numpy()*255).astype(np.uint8))   
                    crop_point_mask = Image.fromarray((segment_mask[0,1].cpu().detach().numpy()*255).astype(np.uint8))
                    crop_pred_point_mask = Image.fromarray((semantic_pred[0,1].sigmoid().cpu().detach().numpy()*255).astype(np.uint8))   
                    dst.paste(crop_segment_mask,(0,0))
                    dst.paste(crop_pred_segment_mask,(200,0))
                    dst.paste(crop_point_mask,(0,200))
                    dst.paste(crop_pred_point_mask,(200,200))
                    dst.save(os.path.join(args.train_save_dir,f'{i}.png'))
                pbar.set_description(f'===Epoch: {epoch} ')
                pbar.update()
        
        if args.local_rank==0:
            print('Start evaluation.....')
            acc,recall,f1 = evaluation(args, model.module, val_loader)
            print(f'ACC/RECALL/F1: {acc}/{recall}/{f1}')
            # torch.save(model.module.state_dict(),os.path.join(args.savedir+'/checkpoints', f"{args.model}_{epoch}.pt"))
            # print(f'Save checkpoint {epoch}')
            if f1>best_f1:
                torch.save(model.module.state_dict(),os.path.join(args.savedir+'/checkpoints', f"{args.model}_best.pt"))
                best_f1 = f1
            writer.add_scalar('eval/acc', acc, epoch)
            writer.add_scalar('eval/recall', recall, epoch)
            writer.add_scalar('eval/f1', f1, epoch)
        dist.barrier()
        model.train()
        sched.step()

if __name__ == '__main__':
    args = get_args()
    setup_seed(20)
    train(args)