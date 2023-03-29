import os
import numpy as np
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
import shutil
import sys 
import json
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
    # Github source: https://github.com/hanson-young/ocr-table-ssd/blob/dev/test_rotate.py
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
    
def generate_scene_map(args,scene_frame_list,i):
    with open(os.path.join(args.dataroot,'label_world_crop',i+'.json'),'r') as jf:
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
        gt_rotate_image = rotate_bound(frame[1],-frame[-1]*180/np.pi)
        sum_for_once = rotate_bound(np.ones((200,200)),-frame[-1]*180/np.pi)
        height, width = gt_rotate_image.shape
        l = int(trans[0]-x_min+pad_size-np.floor(width/2))
        r = int(trans[0]-x_min+pad_size+np.ceil(width/2))
        d = int(trans[1]-y_min+pad_size-np.floor(height/2))
        u = int(trans[1]-y_min+pad_size+np.ceil(height/2))
        pred_scene_map[d:u,l:r] += pred_rotate_image
        gt_scene_map[d:u,l:r] += gt_rotate_image
        sum_num_map[d:u,l:r] += sum_for_once

    sum_num_map[sum_num_map==0] = 1
    pred_scene_map /= sum_num_map
    gt_scene_map /= sum_num_map

    pred_scene_map = np.clip(pred_scene_map* 255,0,255)
    gt_scene_map = np.clip(gt_scene_map * 255,0,255)
    return pred_scene_map.astype(np.uint8), gt_scene_map.astype(np.uint8)

def inference(args):

    # ============== 
    save_dir_single = f'./{args.savedir}/inference/single'
    create_directory(f'./{args.savedir}/inference/single/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/inference/single/pred_mask',delete=True)

    save_dir_multi = f'./{args.savedir}/inference/multi'
    create_directory(f'./{args.savedir}/inference/multi/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/inference/multi/pred_mask',delete=True)

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

    _, val_loader = segmentation_dataset(args, data_conf, multi_GPU=False, is_test=True)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.seg_checkpoint_dir,map_location='cpu'))
    model.cuda()
    model.eval()
    
    pre_scene = None
    scene_frame_list = []
    frame_idx = 0
    with torch.no_grad():
        with tqdm(total=len(val_loader), unit='img') as pbar:
            for i, data in enumerate(val_loader):
                scene_name = data['name'][0]
                if pre_scene is None:
                    pre_scene = scene_name
                    pass
                elif scene_name!=pre_scene or i==len(val_loader)-1:
                    pred_scene_map, gt_scene_map = generate_scene_map(args,scene_frame_list,pre_scene)
                    scene_frame_list = []
                    Image.fromarray(pred_scene_map).convert('RGB').save(os.path.join(save_dir_multi,'pred_mask',f'{pre_scene}.png'))
                    Image.fromarray(gt_scene_map).convert('RGB').save(os.path.join(save_dir_multi,'gt_mask',f'{pre_scene}.png'))
                    pre_scene = scene_name
                    frame_idx = 0
                    
                segment_mask = data['segment_mask'].unsqueeze(1).cuda()

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

                semantic_pred = semantic_pred[0,0].sigmoid().cpu().detach().numpy()
                scene_frame_list.append([(semantic_pred),\
                            (((segment_mask[0,0].cpu().detach().numpy()*255)>200)).astype(np.uint8),\
                            data['car_trans'][0].cpu().detach().numpy()*4,\
                            data['yaw_pitch_roll'][0,0].cpu().detach().numpy()])
                
                pbar.update()
                
                Image.fromarray((segment_mask[0,0].cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(save_dir_single,'gt_mask',f'{scene_name}_{frame_idx}.png'))
                Image.fromarray((semantic_pred*255).astype(np.uint8)).save(os.path.join(save_dir_single,'pred_mask',f'{scene_name}_{frame_idx}.png'))
                frame_idx += 1

def evaluation(args, model, val_loader):
    
    def calculate_scores(gt_points,pred_points):
        gt_tree = cKDTree(gt_points)
        if len(pred_points):
            pred_tree = cKDTree(pred_points)
        else:
            return 0,0,0
        for c_i,thre in enumerate([5]):
            dis_gt2pred,_ = pred_tree.query(gt_points, k=1)
            dis_pred2gt,_ = gt_tree.query(pred_points, k=1)
            recall = len([x for x in dis_gt2pred if x<thre])/len(dis_gt2pred)
            acc = len([x for x in dis_pred2gt if x<thre])/len(dis_pred2gt)
        r_f = 0
        if acc*recall:
            r_f = 2*recall * acc / (acc+recall)
        return acc, recall, r_f

    def pixel_eval_metric(pred_mask,gt_mask):
        def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

        gt_points = tuple2list(np.where(gt_mask!=0))
        pred_points = tuple2list(np.where(pred_mask!=0))

        return calculate_scores(gt_points,pred_points)

    def get_score(pred_mask,gt_mask):
        output = []
        pred_mask = (pred_mask > 0.2)
        pred_skeleton = skeletonize(pred_mask, method='lee')
        gt_mask = (gt_mask > 0.1)
        gt_skeleton = skeletonize(gt_mask, method='lee')
        output.append(pixel_eval_metric(pred_skeleton,gt_skeleton))
        return output

    model.eval()
    pre_scene = None
    scene_frame_list = []
    frame_idx = 0
    score_list = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), unit='img') as pbar:
            for i, data in enumerate(val_loader):
                scene_name = data['name'][0]
                if pre_scene is None:
                    pre_scene = scene_name
                    pass
                elif scene_name!=pre_scene or i==len(val_loader)-1:
                    pred_scene_map, gt_scene_map = generate_scene_map(args,scene_frame_list,pre_scene)
                    scene_frame_list = []
                    Image.fromarray(pred_scene_map).convert('RGB').save(os.path.join(args.eval_save_dir,'multi/pred_mask',f'{pre_scene}.png'))
                    Image.fromarray(gt_scene_map).convert('RGB').save(os.path.join(args.eval_save_dir,'multi/gt_mask',f'{pre_scene}.png'))
                    pre_scene = scene_name
                    frame_idx = 0

                    score_list.extend(get_score(pred_scene_map,gt_scene_map))
                    
                segment_mask = data['segment_mask'].unsqueeze(1).cuda()

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

                semantic_point_pred = semantic_pred[0,1].sigmoid().cpu().detach().numpy()
                semantic_pred = semantic_pred[0,0].sigmoid().cpu().detach().numpy()
                scene_frame_list.append([(semantic_pred),\
                            (((segment_mask[0,0].cpu().detach().numpy()*255)>200)).astype(np.uint8),\
                            data['car_trans'][0].cpu().detach().numpy()*4,\
                            data['yaw_pitch_roll'][0,0].cpu().detach().numpy()])
                
                pbar.update()
                
                Image.fromarray((segment_mask[0,0].cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(args.eval_save_dir,'single/gt_mask',f'{scene_name}_{frame_idx}.png'))
                Image.fromarray((data['initial_candidate_mask'][0].cpu().detach().numpy()*255).astype(np.uint8)).save(os.path.join(args.eval_save_dir,'single/gt_point_mask',f'{scene_name}_{frame_idx}.png'))
                Image.fromarray((semantic_pred*255).astype(np.uint8)).save(os.path.join(args.eval_save_dir,'single/pred_mask',f'{scene_name}_{frame_idx}.png'))
                Image.fromarray((semantic_point_pred*255).astype(np.uint8)).save(os.path.join(args.eval_save_dir,'single/pred_point_mask',f'{scene_name}_{frame_idx}.png'))
                frame_idx += 1
    return sum([x[0] for x in score_list])/(len(score_list)+1e-7),sum([x[1] for x in score_list])/(len(score_list)+1e-7),sum([x[2] for x in score_list])/(len(score_list)+1e-7)

if __name__ == '__main__':
    args = get_args()
    setup_seed(20)
    inference(args)