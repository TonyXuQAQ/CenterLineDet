import os
import numpy as np
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter
import argparse
from PIL import Image, ImageDraw
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from inference import evaluate

import sys
sys.path.append('..')
from config.get_args import get_args
from data.dataset import centerlinedet_dataset
from model.detr import build_model


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
    
    data_conf = {
        'num_channels': 2,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    # =====================
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(f'cuda:{args.local_rank}')
    train_loader, train_sampler, val_loader = centerlinedet_dataset(args, data_conf)
    CenterLineDet, criterion_DETR = build_model(args)
    CenterLineDet.cuda()
    CenterLineDet = torch.nn.parallel.DistributedDataParallel(CenterLineDet, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    opt = torch.optim.AdamW(CenterLineDet.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sched = StepLR(opt, 20, 0.25)
    if args.local_rank==0:
        args.train_save_dir = f'./{args.savedir}/train'
        args.val_save_dir = f'./{args.savedir}/val'
        create_directory(f'./{args.savedir}/tensorboard',delete=True)
        create_directory(f'./{args.savedir}/tensorboard_past')
        create_directory(f'./{args.savedir}/train',delete=True)
        create_directory(f'./{args.savedir}/val',delete=True)
        create_directory(f'./{args.savedir}/checkpoints')

        writer = SummaryWriter(args.savedir+'/tensorboard')
    # ====================
    best_f1 = 0
    for epoch in range(args.nepochs):
        loss_prob_list = []
        loss_coord_list = []
        loss_seg_list = []
        CenterLineDet.train()
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), unit='img') as pbar:
            for i, data in enumerate(train_loader):
                    
                # Calculate labels
                feature_tensor, gt_seg, gt_probs, gt_coords, gt_ahead_segment, list_len = data
                feature_tensor, gt_seg, gt_probs, gt_coords, gt_ahead_segment = feature_tensor.cuda(), gt_seg.cuda(), gt_probs.cuda(), gt_coords.cuda(), gt_ahead_segment.cuda()
                
                outputs = CenterLineDet(feature_tensor)
                    
                # Calculate labels
                targets = [{'labels':gt_probs[x,:list_len[x]],'boxes':gt_coords[x,:list_len[x]], 'masks':gt_seg[x],'ahead_mask':gt_ahead_segment[x,:list_len[x]]} for x in range(gt_probs.shape[0])]
                
                # Loss function 
                # loss_seg_dict = loss_fn(semantic_pred, semantic_gt)
                loss_dict = criterion_DETR(outputs,targets)
                try:
                    loss_prob, loss_coord, loss_seg = loss_dict['loss_ce'], loss_dict['loss_bbox'] * 5, loss_dict['loss_ahead_seg']
                    final_loss = loss_prob + loss_coord + loss_seg
                    loss_coord_list.append(loss_coord.item())
                except:
                    loss_prob, loss_seg = loss_dict['loss_ce'], loss_dict['loss_ahead_seg']
                    final_loss = loss_prob + loss_seg
                loss_seg_list.append(loss_seg.item())
                loss_prob_list.append(loss_prob.item())
                opt.zero_grad()
                final_loss.backward()
                opt.step()
                
                if i%10==0:
                    dst = Image.new('RGB',(args.ROI_half_length*2*3+5,args.ROI_half_length*2*2+5))
                    crop_segment_mask = Image.fromarray((feature_tensor[0,0].cpu().detach().numpy()*255).astype(np.uint8))
                    crop_history = Image.fromarray((feature_tensor[0,-1].cpu().detach().numpy()*255).astype(np.uint8))
                    gt_ahead_segment = Image.fromarray(np.clip((torch.sum(gt_ahead_segment[0],dim=0)*255).cpu().detach().numpy(),0,255).astype(np.uint8))
                    pred_ahead_segment = F.interpolate(outputs['pred_ahead_masks'],  size=(args.ROI_half_length*2,args.ROI_half_length*2), mode='bilinear')
                    pred_ahead_segment = Image.fromarray(np.clip((torch.sum(pred_ahead_segment[0],dim=0)*255).cpu().detach().numpy(),0,255).astype(np.uint8))
                    dst.paste(crop_segment_mask,(0,0))
                    dst.paste(crop_history,(args.ROI_half_length*2,0))
                    dst.paste(gt_ahead_segment,(args.ROI_half_length*2*2,0))
                    dst.paste(pred_ahead_segment,(args.ROI_half_length*2*2,args.ROI_half_length*2))
                    draw = ImageDraw.Draw(dst)
                    for ii in range(3):
                        draw.ellipse([ii*args.ROI_half_length*2+args.ROI_half_length-1,args.ROI_half_length-1,(ii)*args.ROI_half_length*2+args.ROI_half_length+1,args.ROI_half_length+1],fill='orange')
                        for kk in range(gt_probs[0,:list_len[0]].shape[0]):
                            v_next = gt_coords[0,:list_len[0]][kk].cpu().detach().numpy()
                            draw.ellipse([ii*args.ROI_half_length*2+args.ROI_half_length-1+(v_next[0]*args.ROI_half_length),args.ROI_half_length-1+(v_next[1]*args.ROI_half_length),\
                                (ii)*args.ROI_half_length*2+args.ROI_half_length+1+(v_next[0]*args.ROI_half_length),args.ROI_half_length+1+(v_next[1]*args.ROI_half_length)],fill='cyan')
                    
                    pred_coord = outputs['pred_boxes'].cpu().detach().numpy()[0]
                    pred_logit = outputs['pred_logits'].cpu().detach().numpy()[0]
                    for jj in range(pred_coord.shape[0]):
                        v = pred_coord[jj]
                        v = [v[0]*args.ROI_half_length+args.ROI_half_length+args.ROI_half_length*2, v[1]*args.ROI_half_length+args.ROI_half_length]
                        if pred_logit[jj][0] < pred_logit[jj][1]:
                            draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='yellow',outline='yellow')
                        else:
                            draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='pink',outline='pink')

                    dst.save(os.path.join(args.savedir,'train',f'{i}.png'))
                        
                pbar.update()
                # break

        if args.local_rank==0:
            # with torch.no_grad():
                print('Start evaluation.....')
                acc,recall,f1 = evaluate(args,data_conf,CenterLineDet.module,val_loader)
                writer.add_scalar('eval/acc', acc, epoch)
                writer.add_scalar('eval/recall', recall, epoch)
                writer.add_scalar('eval/f1', f1, epoch)
                # torch.save({'model':CenterLineDet.module.state_dict(),
                #             'schedulaer':sched,
                #             'opt':opt,
                #             'epoch':epoch},
                #     os.path.join(args.savedir+'/checkpoints', f"detr_{args.training_mode}_{epoch}.pt"))
                if f1 > best_f1:
                    torch.save({'model':CenterLineDet.module.state_dict(),
                            'schedulaer':sched,
                            'opt':opt,
                            'epoch':epoch},
                    os.path.join(args.savedir+'/checkpoints', f"center_{args.model}_best.pt"))
                    best_f1 = f1
                print(f'Acc/Recall/f1:{acc}/{recall}/{f1}')
                print(f'============ Save checkpoint: {epoch}')
        dist.barrier()
        sched.step()
        CenterLineDet.train()
        
if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False 
    args = get_args()
    setup_seed(20)
    train(args)
