import os
from matplotlib.pyplot import ylabel
import numpy as np

import torch
import cv2
from PIL import Image
import json
import random
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from torch.utils.data import Dataset
from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from .lidar import get_lidar_data
from .image import normalize_img, img_transform
from .utils import label_onehot_encoding
from model.voxel import pad_or_trim_to_np
from .env import Environment

with open('../data/split.json','r') as jf:
    split_data = json.load(jf)
TRAIN_SCENES_1, TRAIN_SCENES_2, VAL_SCENES, TEST_SCENES = split_data['train_1'],split_data['train_2'],split_data['val'],split_data['test']

class HDMapNetDataset(Dataset):
    def __init__(self,args, nusc, data_conf, is_train=False, train_set_selection=False):
        super(HDMapNetDataset, self).__init__()
        self.args = args
        version = args.version
        dataroot = args.dataroot
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
        canvas_h = int(patch_h / data_conf['ybound'][2])
        canvas_w = int(patch_w / data_conf['xbound'][2])
        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = nusc
        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()
        self.is_depth = data_conf.get('is_depth', False)
        self.previous_frame_scene_name = None

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        for i,samp in enumerate(samples):
            samples[i]['scene_name'] = self.nusc.get('scene', samp['scene_token'])['name']
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_name'], x['timestamp']))


        return samples

    def get_lidar(self, rec):
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
        lidar_data = lidar_data.transpose(1, 0)
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)
    

    def get_info(self, rec,idx):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_pose = ego_pose['translation'][:2]
        rotation = Quaternion(ego_pose['rotation'])
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        return [location,patch_box,patch_angle,rec,idx]

    def make_transform_matrix(self,record):
        """
        Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
        """
        my_transform = np.eye(4)
        my_transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
        my_transform[:3, 3] = np.array(record['translation'])
        return my_transform

    def get_sensor_transform(self,rec):
        sample_data = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        
        # Load sensor transform data
        sensor = self.nusc.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])

        # Load ego pose data
        pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        car_trans = [sensor['translation'][i] + pose['translation'][i] for i in range(3)]
        pos_rotation = Quaternion(pose['rotation'])
        sensor_rotation = Quaternion(pose['rotation'])
        yaw_pitch_roll = [pos_rotation.yaw_pitch_roll[i] + sensor_rotation.yaw_pitch_roll[i] for i in range(3)]

        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_depth(self, samp, cam_name, resize_dims):
        depth_name = os.path.join(
            self.nusc.dataroot, 
            samp['filename'].replace(cam_name, cam_name+'_DEPTH_GT').replace('.jpg', '.png')
        )
        depth = np.array(
                cv2.imread(depth_name, -1) / 256.0, dtype=np.float32
        )
        depth[depth > 50] = 50
        resized_depth = cv2.resize(depth, (resize_dims[0], resize_dims[1]), interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.from_numpy(resized_depth).float()
        return depth_tensor

    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []
        depths = []

        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)

            resize, resize_dims = self.sample_augmentation()
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

            if self.is_depth:
                depth = self.get_depth(
                    samp, cam, resize_dims
                )
                depths.append(depth)

            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
        
        img = torch.stack(imgs)
        trans = torch.stack(trans)
        rots = torch.stack(rots)
        intrins= torch.stack(intrins)
        post_trans = torch.stack(post_trans)
        post_rots = torch.stack(post_rots)
        return img, trans, rots, intrins, post_trans, post_rots



    def get_centerline_map(self, rec,idx, env):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_pose = ego_pose['translation'][:2]
        rotation = Quaternion(ego_pose['rotation'])
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        save_scene_name = None
        if (rec['scene_name'] != self.previous_frame_scene_name and self.previous_frame_scene_name is not None) or (idx==len(self.samples)-1):
            save_scene_graph = 1
            save_scene_name = self.previous_frame_scene_name
            if idx==len(self.samples)-1:
                save_scene_name = rec['scene_name']
                save_scene_graph = 2
        else:
            save_scene_graph = 0
        env.get_centerline_mask(location,patch_box,patch_angle,rec,idx,save_scene_graph,save_scene_name)
        self.previous_frame_scene_name = rec['scene_name']


    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors = self.get_vectors(rec)

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors

# ====================================================================================                                                   

class PrepareDataset(HDMapNetDataset):
    def __init__(self, args,nusc, data_conf, is_train):
        super(PrepareDataset, self).__init__(args, nusc, data_conf, is_train)
        self.is_train = is_train
        self.env = Environment(args,self.canvas_size,self.patch_size)

    def __getitem__(self, idx):
        idx = idx  
        rec = self.samples[idx]
        self.get_centerline_map(rec,idx,self.env)
        return idx

class SegmentationDataset(HDMapNetDataset):
    def __init__(self,args, nusc, data_conf, is_train, is_test=False, train_set_selection='all', test_and_val=False):
        super(SegmentationDataset, self).__init__(args,nusc, data_conf, is_train)
        self.is_train = is_train
        self.train_set_selection = train_set_selection
        data_list = os.listdir('../dataset/label_trainval')
        if is_train:
            if train_set_selection=='all':
                data_list = [x for x in data_list if x.split('_')[0] in TRAIN_SCENES_1+TRAIN_SCENES_2]
            elif train_set_selection=='train1':
                data_list = [x for x in data_list if x.split('_')[0] in TRAIN_SCENES_1]
            elif train_set_selection=='train2':
                data_list = [x for x in data_list if x.split('_')[0] in TRAIN_SCENES_2]
            else:
                raise Exception('Invalid value of train_set_selection...')
        elif test_and_val:
            data_list = [x for x in data_list if x.split('_')[0] in TEST_SCENES+VAL_SCENES]
        elif is_test:
            data_list = [x for x in data_list if x.split('_')[0] in TEST_SCENES]
        else:
            data_list = [x for x in data_list if x.split('_')[0] in VAL_SCENES]
        data_list.sort()
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample = np.load(f'../dataset/label_trainval/{data}',allow_pickle=True)
        
        rec = sample['rec'].item()
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        output = {
            'imgs':imgs,
            'trans':trans, 
            'rots':rots, 
            'intrins':intrins, 
            'post_trans':post_trans, 
            'post_rots':post_rots, 
            'lidar_data':lidar_data, 
            'lidar_mask':lidar_mask, 
            'car_trans':car_trans, 
            'yaw_pitch_roll':yaw_pitch_roll,
            'segment_mask':sample['segment_mask'],
            'initial_candidate_mask':sample['initial_candidate_mask'],
            'name':data.split('_')[0]
        }
        return output

def collate_fn(batch):
    output = {
            'imgs':[],
            'trans':[], 
            'rots':[], 
            'intrins':[], 
            'post_trans':[], 
            'post_rots':[], 
            'lidar_data':[], 
            'lidar_mask':[], 
            'car_trans':[], 
            'yaw_pitch_roll':[],
            'segment_mask':[],
            'initial_candidate_mask':[],
            'name':[]
        }
    for data in batch:
        for k,v in data.items():
            if isinstance(v,np.ndarray) and 'name' not in k:
                output[k].append(torch.FloatTensor(v).unsqueeze(0))
            elif k=='name' or k=='info':
                output[k].append(v)
            else:
                output[k].append(v.unsqueeze(0))
    for k,v in output.items():
        if k!='name' and k!='info':
            output[k] = torch.cat(output[k],dim=0)
    return output

class  CenterLineDetInferenceDataset(HDMapNetDataset):
    def __init__(self,args, nusc, data_conf, is_test=False, vis=False):
        super(CenterLineDetInferenceDataset, self).__init__(args,nusc, data_conf)
        data_list = os.listdir('../dataset/label_trainval')
        if is_test:
            data_list = [x for x in data_list if x.split('_')[0] in TEST_SCENES]
            data_list.sort()
            self.data_list = data_list#[:410]
        else:
            data_list = [x for x in data_list if x.split('_')[0] in VAL_SCENES]
            data_list.sort()
            self.data_list = data_list#[:410]
        
        if vis:
            data_list = [x for x in data_list if x.split('_')[0] == 'scene-0054']
            data_list.sort()
            self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample = np.load(f'../dataset/label_trainval/{data}',allow_pickle=True)
        
        rec = sample['rec'].item()
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        cam_trans, cam_yaw_pitch_roll = self.get_sensor_transform(rec)
        info = self.get_info(rec,idx)
        
        pred_candidate_init = sample['initial_candidate_mask']
        output = {
            'imgs':imgs,
            'trans':trans, 
            'rots':rots, 
            'intrins':intrins, 
            'post_trans':post_trans, 
            'post_rots':post_rots, 
            'lidar_data':lidar_data, 
            'lidar_mask':lidar_mask, 
            'car_trans':car_trans, 
            'yaw_pitch_roll':yaw_pitch_roll,
            'cam_trans':cam_trans,
            'cam_yaw_pitch_roll':cam_yaw_pitch_roll,
            'segment_mask':sample['segment_mask'],
            'initial_candidate_mask':sample['initial_candidate_mask'],
            'name':data.split('_')[0],
            'info':info,
            'pred_initial_candidate_mask':pred_candidate_init
        }
        return output

def DAgger_collate_fn(batch):
    output = {
            'imgs':[],
            'trans':[], 
            'rots':[], 
            'intrins':[], 
            'post_trans':[], 
            'post_rots':[], 
            'lidar_data':[], 
            'lidar_mask':[], 
            'car_trans':[], 
            'yaw_pitch_roll':[],
            'cam_trans':[],
            'cam_yaw_pitch_roll':[],
            'segment_mask':[],
            'initial_candidate_mask':[],
            'name':[],
            'info':[],
            'pred_initial_candidate_mask':[]
        }
    for data in batch:
        for k,v in data.items():
            if isinstance(v,np.ndarray) and 'name' not in k:
                output[k].append(torch.FloatTensor(v).unsqueeze(0))
            elif k=='name' or k=='info':
                output[k].append(v)
            else:
                output[k].append(v.unsqueeze(0))

    for k,v in output.items():
        if k!='name' and k!='info':
            output[k] = torch.cat(output[k],dim=0)
    return output


class DETRSamplerDataset(HDMapNetDataset):
    def __init__(self,args, nusc,data_conf, is_train, train_set_selection):
        super(DETRSamplerDataset, self).__init__(args,nusc, data_conf, is_train,train_set_selection)
        self.is_train = is_train
        
        data_list = os.listdir('../dataset/label_trainval')
        if is_train:
            if train_set_selection=='all':
                data_list = [x for x in data_list if x.split('_')[0] in TRAIN_SCENES_1+TRAIN_SCENES_2]
            elif train_set_selection=='train1':
                data_list = [x for x in data_list if x.split('_')[0] in TRAIN_SCENES_1]
            elif train_set_selection=='train2':
                data_list = [x for x in data_list if x.split('_')[0] in TRAIN_SCENES_2]
            else:
                raise Exception('Invalid value of train_set_selection...')
        else:
            data_list = [x for x in data_list if x.split('_')[0] in VAL_SCENES]
        data_list.sort()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample = np.load(f'../dataset/label_trainval/{data}',allow_pickle=True)
        rec = sample['rec'].item()
        info = self.get_info(rec,idx)
        output = {
            'segment_mask':sample['segment_mask'],
            'info': info,
            'name':data.split('_')[0]
        }
        return output

def detr_collate_fn(batch):
    data = batch[0]
    for k,v in data.items():
        if isinstance(v,np.ndarray) and 'v_' not in k:
            data[k] = torch.FloatTensor(v).unsqueeze(0)
        elif k=='info' or k=='name':
            data[k] = v
        elif 'v_' in k:
            data[k] = np.array(v, dtype="object")
        else:
            data[k] = v.unsqueeze(0)
    return data

class CenterLineDetDataset(Dataset):
    def __init__(self,args,is_train):
        self.args = args
        self.is_train = is_train
        root_dir = f'./{args.savedir}/samples_detr_train/'
        dir_list = [root_dir+directory for directory in os.listdir(root_dir) if os.path.isdir(root_dir+directory)]
        data_list = []
        if args.training_mode=='BC':
            for dir in [f'./{args.savedir}/samples_detr_train/BC']:
                sub_data_list = os.listdir(dir)
                data_list.extend([os.path.join(dir,x) for x in sub_data_list])
            self.data_list = data_list[:len(data_list)]
        else:
            raise Exception('Wrong mode...')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):
        data = self.data_list[idx]
        sample = np.load(data,allow_pickle=True)

        rot_index = np.random.randint(0,4)
        feature_tensor = np.rot90(sample['feature_tensor'][0],rot_index,[1,2]).copy()
        gt_segment_ahead = np.rot90(sample['gt_segment_ahead'],rot_index,[0,1]).copy()

        theta = rot_index * np.pi / 2
        R = np.array([[np.cos(theta),np.sin(theta)],[np.sin(-theta),np.cos(theta)]])
        gt_coords = R.dot(sample['gt_coords'].T).T
        output_tensor = torch.FloatTensor(feature_tensor)[[0,-1]]/255.0
        list_len = sample['list_len'] 
        # data torch
        gt_seg, gt_probs, gt_coords, gt_segment_ahead = \
            torch.FloatTensor(feature_tensor)/255.0,\
            torch.LongTensor(sample['gt_probs']).reshape(self.args.num_queries),\
            torch.FloatTensor(gt_coords).reshape(self.args.num_queries,2),\
            torch.FloatTensor(gt_segment_ahead).permute(2,0,1)
             
        return output_tensor, gt_seg, gt_probs, gt_coords, gt_segment_ahead, list_len


# ====================================== 
def prepare_dataset(args,data_conf):
    print('Loading nuscenes devkit......')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print('Finish loading nuscenes devkit......')
    train_dataset = PrepareDataset(args, nusc,data_conf, is_train=True)
    val_dataset = PrepareDataset(args, nusc, data_conf, is_train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers)
    return train_loader, val_loader

def segmentation_dataset(args, data_conf, multi_GPU=True, is_test=False, train_set_selection='all', test_and_val = False):
    print('Loading nuscenes devkit......')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print('Finish loading nuscenes devkit......')
    train_dataset = SegmentationDataset(args,nusc, data_conf, is_train=True,train_set_selection=train_set_selection)
    val_dataset = SegmentationDataset(args, nusc, data_conf, is_train=False,is_test=is_test,train_set_selection=train_set_selection,test_and_val=test_and_val)
    if multi_GPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.bsz,num_workers=args.nworkers,pin_memory=True,sampler=train_sampler,collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.nworkers,collate_fn=collate_fn)
        print(f'Training data: {len(train_dataset)} || Validation data: {len(val_dataset)}')
        return train_dataloader,train_sampler, val_dataloader
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers,collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.nworkers,collate_fn=collate_fn)
        print(f'Training data: {len(train_dataset)} || Validation data: {len(val_dataset)}')
        return train_dataloader, val_dataloader

def centerlinedet_sampler_dataset(args, data_conf,train_set_selection='train2'):
    print('Loading nuscenes devkit......')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print('Finish loading nuscenes devkit......')
    train_dataset = DETRSamplerDataset(args, nusc,data_conf, is_train=True,train_set_selection=train_set_selection)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers)
    return train_loader, nusc

def centerlinedet_dataset(args, data_conf, multi_GPU=True):
    print('Loading nuscenes devkit......')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print('Finish loading nuscenes devkit......')
    train_dataset = CenterLineDetDataset(args, is_train=True)
    val_dataset = CenterLineDetInferenceDataset(args, nusc, data_conf)
    if multi_GPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.bsz,num_workers=6,pin_memory=True,sampler=train_sampler)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.nworkers,collate_fn=DAgger_collate_fn)
        print(f'Training data: {len(train_dataset)} || Validation data: {len(val_dataset)}')
        return train_dataloader, train_sampler, val_dataloader
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, num_workers=args.nworkers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers,collate_fn=DAgger_collate_fn)
    print(f'Training data: {len(train_dataset)} || Validation data: {len(val_dataset)}')
    return train_dataloader, val_dataloader

def inference_centerlinedet_dataset(args, data_conf, vis=False):
    print('Loading nuscenes devkit......')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print('Finish loading nuscenes devkit......')
    val_dataset = CenterLineDetInferenceDataset(args, nusc, data_conf, is_test=True, vis=vis)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers,collate_fn=DAgger_collate_fn)
    print(f'Inference data: {len(val_dataset)}')
    return  val_dataloader

