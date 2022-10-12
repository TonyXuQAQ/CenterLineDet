import os
import numpy as np
import sys
import argparse
from PIL import Image
import torch
from tqdm import tqdm
import cv2
import json
import sys 
import shutil
sys.path.append('..')
from config.get_args import get_args
from data.dataset import segmentation_dataset
from model import get_model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h,w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW,nH))
    

def inference(args):

    def generate_scene_map(scene_frame_list,scene_name):
        with open(os.path.join(args.dataroot,'label_world_crop',scene_name+'.json'),'r') as jf:
            crop_info = json.load(jf)
        x_min, x_max, y_min, y_max = crop_info['x_min'], crop_info['x_max'], crop_info['y_min'], crop_info['y_max']
        pad_size = 0
        pred_scene_map = np.zeros((y_max-y_min+1+pad_size*2,x_max-x_min+1+pad_size*2))
        gt_scene_map = np.zeros((y_max-y_min+1+pad_size*2,x_max-x_min+1+pad_size*2))
        sum_num_map = np.zeros((y_max-y_min+1+pad_size*2,x_max-x_min+1+pad_size*2))
        for ii, frame in enumerate(scene_frame_list):
            trans = frame[-2]
            # rot_mat = cv2.getRotationMatrix2D([100,100], -frame[-1]*180/np.pi, 1.0)
            pred_rotate_image = rotate_bound(frame[0],-frame[-1]*180/np.pi)
            gt_rotate_image = rotate_bound(frame[2],-frame[-1]*180/np.pi)
            sum_for_once = rotate_bound(np.ones((200,200)),-frame[-1]*180/np.pi)
            height, width = gt_rotate_image.shape
            l = int(trans[0]-x_min+pad_size-np.floor(width/2))
            r = int(trans[0]-x_min+pad_size+np.ceil(width/2))
            d = int(trans[1]-y_min+pad_size-np.floor(height/2))
            u = int(trans[1]-y_min+pad_size+np.ceil(height/2))
            try:
                pred_scene_map[d:u,l:r] += pred_rotate_image
                gt_scene_map[d:u,l:r] += gt_rotate_image
                sum_num_map[d:u,l:r] += sum_for_once
            except:
                print(l,r,d,u)
                print(x_min,x_max,y_min,y_max)
                print(trans)
                print(pred_rotate_image.shape)
                print(frame[0]).shape
                raise Exception

        sum_num_map[sum_num_map==0] = 1
        pred_scene_map /= sum_num_map
        gt_scene_map /= sum_num_map

        
        
        for ii, frame in enumerate(scene_frame_list):
            trans = frame[-2]
            rot_mat = cv2.getRotationMatrix2D([trans[0]-x_min+pad_size,trans[1]-y_min+pad_size],frame[-1]*180/np.pi,1.0)
            rotate_gt = cv2.warpAffine(gt_scene_map,rot_mat,gt_scene_map.shape[1::-1],flags=cv2.INTER_LINEAR)
            rotate_pred = cv2.warpAffine(pred_scene_map,rot_mat,gt_scene_map.shape[1::-1],flags=cv2.INTER_LINEAR)
            Image.fromarray(255*rotate_gt[int(trans[1]-y_min+pad_size-100+0.5):int(trans[1]-y_min+pad_size+100+0.5),int(trans[0]-x_min+pad_size-100+0.5):int(trans[0]-x_min+pad_size+100+0.5)]).convert('RGB').save(os.path.join(args.save_dir_fusion,'gt_mask',f'{scene_name}_{ii}.png'))
            rotate_pred = rotate_pred[int(trans[1]-y_min+pad_size-100+0.5):int(trans[1]-y_min+pad_size+100+0.5),int(trans[0]-x_min+pad_size-100+0.5):int(trans[0]-x_min+pad_size+100+0.5)]
            Image.fromarray(rotate_pred*255).convert('RGB').save(os.path.join(args.save_dir_fusion,'pred_mask',f'{scene_name}_{ii}.png'))
            Image.fromarray((rotate_pred*frame[1]*255).astype(np.uint8)).save(os.path.join(args.save_dir_fusion,'pred_initial_candidate',f'{scene_name}_{ii}.png'))
            
        pred_scene_map = np.clip(pred_scene_map * 255,0,255)
        gt_scene_map = np.clip(gt_scene_map * 255,0,255)
        return pred_scene_map.astype(np.uint8), gt_scene_map.astype(np.uint8)

    # ============== 
    args.save_dir_fusion = f'./{args.savedir}'
    create_directory(f'./{args.savedir}/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/pred_fuse',delete=True)
    create_directory(f'./{args.savedir}/gt_fuse',delete=True)
    create_directory(f'./{args.savedir}/pred_initial_candidate',delete=True)
    create_directory(f'./{args.savedir}/gt_initial_candidate',delete=True)

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
    }

    train_loader, val_loader = segmentation_dataset(args, data_conf, multi_GPU=False,train_set_selection='train2', test_and_val=True)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.seg_checkpoint_dir,map_location='cpu'))
    model.cuda()
    model.eval()
           
    
    if 'all' in args.savedir:
        pre_scene = None
        scene_frame_list = []
        frame_index = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), unit='img') as pbar:
                for i, data in enumerate(val_loader):
                    scene_name = data['name'][0]
                    if pre_scene is None:
                        pre_scene = scene_name
                        pass
                    elif scene_name!=pre_scene:
                        pred_scene_map, gt_scene_map = generate_scene_map(scene_frame_list,pre_scene)

                        Image.fromarray(pred_scene_map).convert('RGB').save(os.path.join(args.save_dir_fusion,'pred_fuse',f'{pre_scene}.png'))
                        Image.fromarray(gt_scene_map).convert('RGB').save(os.path.join(args.save_dir_fusion,'gt_fuse',f'{pre_scene}.png'))
                        scene_frame_list = []
                        pre_scene = scene_name
                        frame_index = 0
                    
                    segment_mask = data['segment_mask'].unsqueeze(1).cuda()
                    initial_candidate_mask = data['initial_candidate_mask'].unsqueeze(1).cuda()

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

                    scene_frame_list.append([(semantic_pred[0,0].sigmoid().cpu().detach().numpy()),\
                                (semantic_pred[0,-1].sigmoid().cpu().detach().numpy()),\
                                (((segment_mask[0,0].cpu().detach().numpy()*255)>200)).astype(np.uint8),\
                                data['car_trans'][0].cpu().detach().numpy()*4,\
                                data['yaw_pitch_roll'][0,0].cpu().detach().numpy()])
                    
                    pbar.update()
                    Image.fromarray((initial_candidate_mask[0,0].cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(args.save_dir_fusion,'gt_initial_candidate',f'{scene_name}_{frame_index}.png'))
                    Image.fromarray((semantic_pred[0,-1].sigmoid().cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(args.save_dir_fusion,'pred_initial_candidate',f'{scene_name}_{frame_index}.png'))
                    frame_index += 1

                    if i==len(val_loader)-1:
                        pred_scene_map, gt_scene_map = generate_scene_map(scene_frame_list,scene_name)

                        Image.fromarray(pred_scene_map).convert('RGB').save(os.path.join(args.save_dir_fusion,'pred_fuse',f'{scene_name}.png'))
                        Image.fromarray(gt_scene_map).convert('RGB').save(os.path.join(args.save_dir_fusion,'gt_fuse',f'{scene_name}.png'))
                        
                        break
    
    if 'all' not in args.savedir:
        pre_scene = None
        scene_frame_list = []
        frame_index = 0
        with torch.no_grad():
            with tqdm(total=len(train_loader), unit='img') as pbar:
                for i, data in enumerate(train_loader):
                    scene_name = data['name'][0]
                    if pre_scene is None:
                        pre_scene = scene_name
                        pass
                    elif scene_name!=pre_scene:
                        pred_scene_map, gt_scene_map = generate_scene_map(scene_frame_list,pre_scene)
                        Image.fromarray(pred_scene_map).convert('RGB').save(os.path.join(args.save_dir_fusion,'pred_fuse',f'{pre_scene}.png'))
                        Image.fromarray(gt_scene_map).convert('RGB').save(os.path.join(args.save_dir_fusion,'gt_fuse',f'{pre_scene}.png'))
                        scene_frame_list = []
                        pre_scene = scene_name
                        frame_index = 0
                    
                    segment_mask = data['segment_mask'].unsqueeze(1).cuda()
                    initial_candidate_mask = data['initial_candidate_mask'].unsqueeze(1).cuda()

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

                    scene_frame_list.append([(semantic_pred[0,0].sigmoid().cpu().detach().numpy()),\
                                (semantic_pred[0,-1].sigmoid().cpu().detach().numpy()),\
                                (((segment_mask[0,0].cpu().detach().numpy()*255)>200)).astype(np.uint8),\
                                data['car_trans'][0].cpu().detach().numpy()*4,\
                                data['yaw_pitch_roll'][0,0].cpu().detach().numpy()])
                    
                    pbar.update()
                    Image.fromarray((segment_mask[0,0].cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(args.save_dir_fusion,'gt_initial_candidate',f'{scene_name}_{frame_index}.png'))
                    Image.fromarray((semantic_pred[0,-1].sigmoid().cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(args.save_dir_fusion,'pred_initial_candidate',f'{scene_name}_{frame_index}.png'))
                    frame_index += 1

                    if i==len(train_loader)-1:
                        pred_scene_map, gt_scene_map = generate_scene_map(scene_frame_list,scene_name)
                        break

if __name__ == '__main__':
    args = get_args()
    setup_seed(20)
    inference(args)
