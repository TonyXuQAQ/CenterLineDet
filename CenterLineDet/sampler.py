import numpy as np
import torch
import os
import random
import json
import cv2
from PIL import Image, ImageDraw
import random
import shutil
import argparse 
from shapely import affinity
from shapely.geometry import LineString, Polygon, box, Point
from typing import Tuple
from tqdm import tqdm

import sys
sys.path.append('..')
from config.get_args import get_args
from data.dataset import centerlinedet_sampler_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)


class FrozenClass():
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True

class LineSegment(FrozenClass):
    def __init__(self,id,vertices,orientation):
        self.id = id
        self.init_IP = None
        self.raw_init_IP = None
        self.end_IP = None
        self.raw_end_IP = None
        self.tracked = False
        self.vertices = vertices.copy()
        self.raw_vertices = vertices.copy()
        self.line = LineString([tuple(x) for x in vertices])
        self.orientation = orientation.copy()
        self.raw_orientation = orientation.copy()
        self.tracking_agent = None
        self._freeze()
    
    def reset(self):
        if self.tracked:
            self.vertices = self.raw_vertices.copy()
            self.line = LineString([tuple(x) for x in self.vertices])
            self.orientation = self.raw_orientation.copy()
            self.tracking_agent = None
            self.tracked = False
            self.init_IP = self.raw_init_IP
            self.end_IP = self.raw_end_IP

    def reverse(self):
        self.init_IP, self.end_IP = self.end_IP, self.init_IP
        self.vertices = self.vertices[::-1]
        self.orientation = self.orientation[::-1]
        self.line = LineString([tuple(x) for x in self.vertices])

class IntersectionPoint(FrozenClass):
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
        self.neighbors = []
        self._freeze()

class Patch_LS_Intersection(FrozenClass):
    '''
    This class records the intersection of patch and a line segment. 
    :param LS_inter (LineString): intersection line in shapely format (after transformation)
    :param LS (LineSegment): the corresponding LineSegment instance (before transformation)
    '''
    def __init__(self,env,LS_inter,LS):
        self.LS_inter = LS_inter
        self.LS = LS
        self.env = env
        self._freeze()
    
    def ego2world(self,shapely_object):
        shapely_object = affinity.scale(shapely_object, xfact=1/self.env.scale['width'], yfact=1/self.env.scale['height'], origin=(0, 0))
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, -self.env.translation['x'], -self.env.translation['y']])
        shapely_object = affinity.rotate(shapely_object, self.env.rotation['angle'], origin=(self.env.rotation['x'],self.env.rotation['y']), use_radians=False)
        return shapely_object

    def initialize_tracking(self,initial_candidate):
        warped_point = self.ego2world(Point(self.LS_inter.coords[1]))
        warped_point = [round(x,4) for x in list(warped_point.coords)[0]]
        if len(initial_candidate)==2:
            if list(initial_candidate[0]['point'].coords)[0][0] > 0.1 and list(initial_candidate[1]['point'].coords)[0][0] < 0.1:
                self.LS.reverse()
                warped_point = self.ego2world(Point(self.LS_inter.coords[-2]))
                warped_point = [round(x,4) for x in list(warped_point.coords)[0]]
        elif len(initial_candidate)==1:
            if initial_candidate[0]['id']==2:
                self.LS.reverse()
                warped_point = self.ego2world(Point(self.LS_inter.coords[-2]))
                warped_point = [round(x,4) for x in list(warped_point.coords)[0]]
        
        index = self.LS.vertices.index(warped_point)
        starting_index = index
        return starting_index

class Environment(FrozenClass):
    def __init__(self,args,canvas_size,patch_size):
        # args
        self.args = args
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        # buffer
        self.agents = []
        self.MAPS_NAMES = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.centerline_maps = {}
        self.historical_map = {}
        self.train_samples = {}
        self.patch_LS_inters = []
        # state
        self.agent_index = 0
        self.patch = None
        self.location = ''
        self.rotation = {}
        self.translation = {}
        self.scale = {}
        # counter
        self.scene_counter = 0
        self.overall_step = 0
        self.sample_counter = 0
        # Network 
        self.pred_segmentation_mask = None
        # Initialization
        self.load_map()
        self.__setup_seed(20)
        self._freeze()

    def __setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def load_map(self):
        '''
        Load the ground-truth map graph from json files
        '''
        for map_name in self.MAPS_NAMES:
            LSs = {}
            IPs = {}
            # Process map data
            with open(f'../data/maps/{map_name}.json','r') as jf:
                json_data = json.load(jf)
            json_segments = json_data['lanes']
            json_intersections = json_data['junctions']
            for json_segment in json_segments:
                segment = LineSegment(json_segment['id'],json_segment['vertices'],json_segment['orientation'])
                LSs[json_segment['id']] = segment
            for json_intersection in json_intersections:
                intersection = IntersectionPoint(json_intersection['id'],json_intersection['x'],json_intersection['y'])
                for LS_id in json_intersection['neighbors']:
                    intersection.neighbors.append(LSs[LS_id])
                    if LSs[LS_id].vertices[0] == [intersection.x,intersection.y]:
                        LSs[LS_id].init_IP = intersection
                        LSs[LS_id].raw_init_IP = intersection
                    elif LSs[LS_id].vertices[-1] == [intersection.x,intersection.y]:
                        LSs[LS_id].end_IP = intersection
                        LSs[LS_id].raw_end_IP = intersection
                    else:
                        raise Exception('ERROR, duplicate intersection point')
                IPs[tuple([json_intersection['y'],json_intersection['x']])] = intersection
            self.centerline_maps[map_name] = {'LSs':LSs,'IPs':IPs}
            self.historical_map[map_name] = []
            self.train_samples[map_name] = []

    def get_patch_coord(self,patch_box: Tuple[float, float, float, float],
                        patch_angle: float = 0.0) -> Polygon:
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

        return patch

    def mask_for_lines(self,lines, mask):
        '''
        Draw lines on a mask

        :param lines (LinsString): input lines
        :param mask (np.array): mask before drawing
        :return mask (np.array): mask after drawing
        '''
        coords = np.asarray(list(lines.coords), np.int32)
        coords = coords.reshape((-1, 2))
        cv2.polylines(mask, [coords], False, color=1, thickness=1)
        return mask

    def mask_for_points(self,points, mask, rad=3):
        '''
        Draw points on a mask

        :param points (point): input points
        :param mask (np.array): mask before drawing
        :param rad (int): rad of the drawn points
        :return mask (np.array): mask after drawing
        '''
        coords = np.asarray(list(points.coords), np.int32)
        coords = coords.reshape((-1, 2))
        cv2.circle(mask, coords[0], rad, 1, -1)
        return mask
    
    def reset_env(self):
        '''
        Clean everything. Called when frame changes.
        '''
        self.agents = []
        self.patch_LS_inters = []
        self.agent_index = 0
        for map_name in self.MAPS_NAMES:
            centerline_map = self.centerline_maps[map_name]
            LSs = centerline_map['LSs']
            for i,s in LSs.items():
                s.reset()

            self.historical_map[map_name] = []

    def update_historical_map(self,location,p1,p2):
        '''
        Update the historical map by appending a new LineString

        :param location (str): map name
        :param p1 (Point): src
        :param p2 (Point): dst
        '''
        map = self.historical_map[location]
        map.append(LineString([tuple(p1),tuple(p2)]))
    
    def update_pred_mask(self,pred_mask):
        '''
        Update predicted mask. Called when frame changes.

        :param pred_mask (tensor, shape=(1,2,ROI_half_length,ROI_half_length): updated predicted mask. 
            First channel is predicted segmentation of the current BEV frame,
            second channel is gt segmentation mask.
        '''
        self.pred_segmentation_mask = pred_mask
        self.reset_env()

    def generate_samples(self,location,patch_box,patch_angle,rec):
        '''
        Main function for BC sampling

        :param location (str): name of the map
        :param patch_box (list): info of the patch box of the current frame
        :param patch_angle (float): rot angle of the patch box
        :param rec (nuscene sample item): the nuscene sample of the current frame
        '''
        # Initialization
        self.patch_LS_inters = []
        self.agent_index = 0

        # Load corresponding map
        centerline_map = self.centerline_maps[location]
        self.location = location
        LSs, IPs = centerline_map['LSs'], centerline_map['IPs']
        
        # Patch processing
        patch_x, patch_y, patch_h, patch_w = patch_box
        canvas_h = self.canvas_size[0]
        canvas_w = self.canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0
        self.patch = self.get_patch_coord(patch_box, patch_angle)
        self.rotation = {'angle':patch_angle,'x':patch_x,'y':patch_y}
        self.translation = {'x':trans_x,'y':trans_y}
        self.scale = {'width':scale_width,'height':scale_height}

        # Generate LineStrings that intersect with the patch (LineStrings within the patch)
        for _,s in LSs.items():
            new_line = s.line.intersection(self.patch)
            if not new_line.is_empty:
                new_line = self.world2ego(new_line)
                if new_line.geom_type == 'MultiLineString':
                    for new_single_line in new_line:
                        self.patch_LS_inters.append(Patch_LS_Intersection(self,new_single_line,s))
                else:
                    self.patch_LS_inters.append(Patch_LS_Intersection(self,new_line,s))
        
        # Calculate labels for transformer
        self.overall_step = 0
        self.get_expert_label()
        # Save samples
        for ii,sample in enumerate(self.train_samples[location]):
            self.sample_counter += 1
            feature_tensor = sample['feature_tensor']
            try:
                gt_probs, gt_coords, gt_segment_ahead, list_len = self.calcualte_label(sample['vt'],sample['v_nexts'],sample['segment_ahead'])
                np.savez(os.path.join(f'{self.args.sample_dir}_train/BC',f'{rec["scene_name"][0]}_{self.sample_counter:011}.npz'),feature_tensor=(feature_tensor.cpu().detach().numpy()*255).astype(np.uint8),\
                        gt_probs=gt_probs,gt_coords=gt_coords,gt_segment_ahead=gt_segment_ahead.astype(np.uint8),list_len=list_len)
            except Exception as e:
                print(e)
                print('Skip invalid sample....')

            # =======================================
            # vis
            if self.sample_counter%1000==0:
                args = self.args
                dst = Image.new('RGB',(args.ROI_half_length*2*2+5,args.ROI_half_length*2*2+5))
                crop_segment_mask = Image.fromarray((feature_tensor[0,0].cpu().detach().numpy()*255).astype(np.uint8))
                crop_point_mask = Image.fromarray((feature_tensor[0,1].cpu().detach().numpy()*255).astype(np.uint8))
                crop_history = Image.fromarray((feature_tensor[0,-1].cpu().detach().numpy()*255).astype(np.uint8))
                gt_segment_ahead = Image.fromarray(np.clip((np.sum(gt_segment_ahead,axis=2)*255),0,255).astype(np.uint8))
                
                dst.paste(crop_segment_mask,(0,0))
                dst.paste(crop_point_mask,(args.ROI_half_length*2,args.ROI_half_length*2))
                dst.paste(crop_history,(args.ROI_half_length*2,0))
                dst.paste(gt_segment_ahead,(0,args.ROI_half_length*2))
                draw = ImageDraw.Draw(dst)
                for ii in range(2):
                    draw.ellipse([ii*args.ROI_half_length*2+args.ROI_half_length-1,args.ROI_half_length-1,(ii)*args.ROI_half_length*2+args.ROI_half_length+1,args.ROI_half_length+1],fill='orange')
                    for kk in range(gt_probs[:list_len].shape[0]):
                        v_next = gt_coords[:list_len][kk]
                        draw.ellipse([ii*args.ROI_half_length*2+args.ROI_half_length-1+(v_next[0]*args.ROI_half_length),args.ROI_half_length-1+(v_next[1]*args.ROI_half_length),\
                            (ii)*args.ROI_half_length*2+args.ROI_half_length+1+(v_next[0]*args.ROI_half_length),args.ROI_half_length+1+(v_next[1]*args.ROI_half_length)],fill='cyan')
                dst.convert('RGB').save(os.path.join(f'{self.args.sample_dir}_vis/BC',f'{rec["scene_name"][0]}_{self.sample_counter}.png'))

        self.train_samples[location] = []
        self.reset_env()



    def ego2world(self,shapely_object):
        '''
        Transform the input object from ego coordinate to world coordinate

        :param shapely object: object in ego coordinate
        :return shapely object: object in world coordinate
        '''
        shapely_object = affinity.scale(shapely_object, xfact=1/self.scale['width'], yfact=1/self.scale['height'], origin=(0, 0))
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, -self.translation['x'], -self.translation['y']])
        shapely_object = affinity.rotate(shapely_object, self.rotation['angle'], origin=(self.rotation['x'],self.rotation['y']), use_radians=False)
        return shapely_object
    
    def world2ego(self,shapely_object):
        '''
        Transform the input object from world coordinate to ego coordinate

        :param shapely object: object in world coordinate
        :return shapely object: object in ego coordinate
        '''
        shapely_object = affinity.rotate(shapely_object, -self.rotation['angle'], origin=(self.rotation['x'],self.rotation['y']), use_radians=False)
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, self.translation['x'], self.translation['y']])
        shapely_object = affinity.scale(shapely_object, xfact=self.scale['width'], yfact=self.scale['height'], origin=(0, 0))
        return shapely_object

    def find_initial_candidates(self,patch_LS_inter):
        '''
        We define initial candidates as the endpoints of cut LineString.

        :param patch_LS_inter: A cut LineString within patch
        :return output: initial candidates 
        '''
        endpoint1_POINT, endpoint2_POINT = patch_LS_inter.LS_inter.boundary
        endpoint1, endpoint2 = list(endpoint1_POINT.coords)[0], list(endpoint2_POINT.coords)[0]
        output = []
        if not (endpoint1[0]>0.1 and endpoint1[1]>0.1 and endpoint1[1]<self.canvas_size[0]-0.01):
            output.append({'id':1,'point':endpoint1_POINT})
        if not (endpoint2[0]>0.1 and endpoint2[1]>0.1 and endpoint2[1]<self.canvas_size[0]-0.01):
            output.append({'id':2,'point':endpoint2_POINT})
        return output

    def update_map(self,src,dst,map):
        '''
        Update the historical map by adding a line starting from src to dst.

        :param src (list, length=2): src point of the added line
        :param dst (list, length=2): src point of the added line
        '''
        src = np.array(src)
        dst = np.array(dst)
        p = src
        d = dst - src
        N = np.max(np.abs(d))
        map[int(src[0]),int(src[1])] = 1
        map[int(dst[0]),int(dst[1])] = 1
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                map[int(round(p[0])),int(round(p[1]))] = 1
                
    def calcualte_label(self,v_current,v_nexts,segments):
        '''
        Calculate coordinate and logit labels
        
        :param v_current: current vertex (center vertex)
        :param v_nexts: vertices in the next step
        :return output_prob: gt logits
        :return output_coord: gt coordinates
        :return list len: length of valid coordinates
        '''
        output_prob = np.ones((self.args.num_queries))
        output_coord = np.ones((self.args.num_queries,2))
        output_segment_ahead = np.zeros((self.args.ROI_half_length*2,self.args.ROI_half_length*2,self.args.num_queries))
        
        for ii,segment in enumerate(segments):
            for jj,src in enumerate(segment[1:]):
                dst = segment[jj]
                self.update_map([int(src[1]-v_current[1]+self.args.ROI_half_length),int(src[0]-v_current[0]+self.args.ROI_half_length)],
                    [int(dst[1]-v_current[1]+self.args.ROI_half_length),int(dst[0]-v_current[0]+self.args.ROI_half_length)],output_segment_ahead[:,:,ii])
                
        list_len = 0

        gt_coords = []
        gt_probs = []
        v_current = np.array(v_current)
        for v_next in v_nexts:
            vector = v_next - v_current
            vector = [(x)/self.args.ROI_half_length for x in vector]
            gt_coords.append(vector)
            gt_probs.append(0)
        gt_coords,gt_probs = np.array(gt_coords),np.array(gt_probs)
        list_len = gt_probs.shape[0]
        if list_len:
            output_prob[:list_len] = gt_probs
            output_coord[:list_len] = gt_coords
        
        return (output_prob), (output_coord), (output_segment_ahead), list_len

    def get_expert_label(self):
        '''
        Generate expert labels by agents exploration
        '''
        # Gnerate agents from initial candidates
        for patch_LS_inter in self.patch_LS_inters:
            if len(patch_LS_inter.LS_inter.coords)>5 and  not patch_LS_inter.LS.tracked:
                # Untracked new instance, initialize an agent to track it
                initial_candidates = self.find_initial_candidates(patch_LS_inter)
                if len(initial_candidates):
                    starting_index = patch_LS_inter.initialize_tracking(initial_candidates)
                    new_agent = Agent(self,patch_LS_inter.LS,self.location,starting_index)
                    self.agents.append(new_agent)
                    patch_LS_inter.LS.tracking_agent = new_agent
        # Process each agent. An agent tracks a centerline instance
        while self.agent_index < len(self.agents) and len(self.agents):
            agent = self.agents[self.agent_index]
            while not agent.cease:
                agent.LS_tracked.tracked = True
                self.overall_step += 1
                if self.overall_step > 100:
                    return
                if agent.step_counter > 20:
                    agent.cease = True
                agent.step()
            self.agent_index += 1
        
class Agent(FrozenClass):
    def __init__(self,env,LS,location,starting_index):
        self.env = env
        self.args = env.args
        # coords with noise (actual coords)
        self.v_current_world = LS.vertices[starting_index] # previous vertex in world
        # coords (gt coords for labeling)
        self.expert_next_world = None # gt vertex in the next step in world
        self.expert_index = starting_index # gt index of the agent
        self.segment_ahead = [] # vertices of ahead segment 
        #
        self.LS_tracked = LS # line segment the agent tracks
        self.LS_mode = True # mode of the agent
        self.removed = False # whether the agent should be removed
        self.cease = False # whether the agent should stop
        self.location = location # map name
        self.step_counter = 0 # number of steps
        
        self._freeze()

    def terminate(self):
        '''
        The agent stops. Should be removed.
        '''
        self.env.agents.remove(self)
        self.LS_tracked.tracking_agent = None
        self.removed = True
        self.cease = True

    def crop_tensor(self,feature_tensor,historical_map,v_current):
        '''
        Crop ROI tensor centering at v_current

        :param feature_tensor: BEV tensor
        :param historical_map: historical map
        :param v_current: coord of current vertex
        :return cropped_tensor: cancatenate ROI tensor (C: BEV segmentation, BEV mask, histotical map) 
        '''
        historical_map = torch.Tensor(historical_map).unsqueeze(0)
        cat_tensor = torch.cat([feature_tensor[0],historical_map.cuda()],dim=0)
        cropped_tensor = torch.zeros(cat_tensor.shape[0],self.args.ROI_half_length*2,self.args.ROI_half_length*2)
        v_current = [int(v_current[1]),int(v_current[0])]
        l,r,d,u = max(0,v_current[1]-self.args.ROI_half_length),min(self.env.canvas_size[0],v_current[1]+self.args.ROI_half_length),\
            max(0,v_current[0]-self.args.ROI_half_length),min(self.env.canvas_size[0],v_current[0]+self.args.ROI_half_length)
        cl,cr,cd,cu = max(0,self.args.ROI_half_length-v_current[1]),min(self.args.ROI_half_length*2,self.env.canvas_size[0]-v_current[1]+self.args.ROI_half_length),\
            max(0,self.args.ROI_half_length-v_current[0]),min(self.args.ROI_half_length*2,self.env.canvas_size[0]-v_current[0]+self.args.ROI_half_length)
        cropped_tensor[:,cd:cu,cl:cr] = cat_tensor[:,d:u,l:r]
        return cropped_tensor.unsqueeze(0).cuda()

    def step(self,whether_sample=True):
        '''
        Agent moves one step ahead.

        :param whether_sample: whether the agent save samples
        '''
        step_size = self.args.step_size if whether_sample else self.args.step_size//2
        def get_frame_historical_map():
            frame_historical_map = np.zeros((200,200), np.uint8)
            for patch_history in self.env.historical_map[self.location]:
                new_line = patch_history.intersection(self.env.patch)
                if not new_line.is_empty:
                    new_line = self.env.world2ego(new_line)
                    if new_line.geom_type == 'MultiLineString':
                        for new_single_line in new_line:
                            frame_historical_map = self.env.mask_for_lines(new_single_line, frame_historical_map)
                    else:
                        frame_historical_map = self.env.mask_for_lines(new_line, frame_historical_map)
            return frame_historical_map
        self.step_counter += 1
        # Intersection-mode
        if not self.LS_mode:
            # Adjacent line segments of the current intersection point
            neighbors = self.LS_tracked.end_IP.neighbors
            # End of the current tracked line segment 
            self.terminate()
            if whether_sample:
                self.env.agent_index -= 1
            v_nexts = []
            v_nexts_noise = []
            segments_ahead = []
            frame_historical_map = get_frame_historical_map()
            for neighbor in neighbors:
                if not neighbor.tracked:
                    # this neighbor line segment is already tracked by an agent
                    if neighbor.tracking_agent is not None and neighbor.tracking_agent in self.env.agents:
                        self.env.agents.remove(neighbor.tracking_agent)
                    # reverse
                    if neighbor.init_IP != self.LS_tracked.end_IP:
                        neighbor.reverse()
                    new_agent = Agent(self.env,neighbor,self.location,0)
                    neighbor.tracked = True
                    neighbor.tracking_agent = new_agent
                    self.env.agents.append(new_agent)
                    new_agent.step(whether_sample=False)
                    if not new_agent.removed:
                        v_nexts_noise.append(new_agent.v_current_world)
                        v_nexts.append(new_agent.expert_next_world)
                        segments_ahead.append(new_agent.segment_ahead)
            # =============== Sampling ===============
            v_current = list(self.env.world2ego(Point(self.v_current_world)).coords)[0]
            for v_next in v_nexts_noise:
                self.env.update_historical_map(self.location,self.v_current_world,v_next)
            v_nexts = [list(self.env.world2ego(Point(x)).coords)[0] for x in v_nexts]
            feature_tensor = self.crop_tensor(self.env.pred_segmentation_mask,frame_historical_map,v_current)
            if len(v_nexts):
                self.env.train_samples[self.location].append({'vt':v_current,'v_nexts':v_nexts,'feature_tensor':feature_tensor,'segment_ahead':segments_ahead})
        # Road-segment-mode
        else:            
            v_current = list(self.env.world2ego(Point(self.v_current_world)).coords)[0]
            frame_historical_map = get_frame_historical_map()
            feature_tensor = self.crop_tensor(self.env.pred_segmentation_mask,frame_historical_map,v_current)
            # find index now
            dds = [np.linalg.norm(np.array(v)-np.array(self.v_current_world)) * self.env.scale['width'] for v in self.LS_tracked.vertices]
            current_index = min(self.expert_index,dds.index(min(dds)))     
            # whether the agent is too closed to the final vertex of the LS
            end_v_flag = False 
            patch_LS_inter = self.LS_tracked.line.intersection(self.env.patch)
            if patch_LS_inter.geom_type=='MultiLineString':
                patch_LS_inter = [x for x in patch_LS_inter if len(list(x.coords))>=3]
                if not len(patch_LS_inter):
                    self.terminate()
                    if whether_sample:
                        self.env.agent_index -= 1
                    return
                patch_LS_inter = patch_LS_inter[-1]
            #
            if patch_LS_inter.is_empty or len(list(patch_LS_inter.coords))<3:
                self.terminate()
                if whether_sample:
                    self.env.agent_index -= 1
                return
            # noise 
            noise = self.args.noise

            last_vertex_within = [round(x,4) for x in list(patch_LS_inter.coords[-2])]
            max_index = self.LS_tracked.vertices.index(last_vertex_within)
            if len([1 for x in dds[current_index:max_index] if x<=step_size])==len(dds[current_index:max_index]):
                # Agent reaches the final vertex of the current LS, enter intersection-mode.
                self.expert_index = max_index
                self.expert_next_world = self.LS_tracked.vertices[self.expert_index]
                expert_next = self.env.world2ego(Point(self.expert_next_world))
                # whether change mode
                end_IP = Point([self.LS_tracked.end_IP.x,self.LS_tracked.end_IP.y])
                if end_IP.within(self.env.patch):
                    self.LS_mode = False
                    end_v_flag = True
                else:
                    self.terminate()
                    if whether_sample:
                        self.env.agent_index -= 1
                    # return
            else:
                # Agent moves along the current centerline instance.
                
                orientation_close_raw = [i for i,v in enumerate(self.LS_tracked.orientation) if i>current_index and dds[i]<step_size]
                orientation_close = [i for i,x in enumerate(orientation_close_raw[1:]) if \
                    min(360-abs(self.LS_tracked.orientation[x]-self.LS_tracked.orientation[orientation_close_raw[0]]),abs(self.LS_tracked.orientation[x]-self.LS_tracked.orientation[orientation_close_raw[0]]))>20 and \
                    x-orientation_close_raw[i]==1]
                if len(orientation_close):
                    # Curve ahead
                    self.expert_index = min(orientation_close[0] + 1+current_index,max_index)
                    self.expert_next_world = self.LS_tracked.vertices[self.expert_index]
                    noise = noise//2
                else:
                    # Straight ahead
                    self.expert_index = [i for i,v in enumerate(self.LS_tracked.vertices) if i>current_index and dds[i]>step_size][0]
                    self.expert_next_world = self.LS_tracked.vertices[self.expert_index]
                expert_next = self.env.world2ego(Point(self.expert_next_world))
                # ============== Sampling ===========
            # segment ahead
            self.segment_ahead = [list(self.env.world2ego(Point(v)).coords)[0] for v in self.LS_tracked.vertices[current_index:self.expert_index]]
            # add noise
            v_next_noise = [x+np.random.uniform(-noise,noise) for x in list(expert_next.coords)[0]]
            v_next_noise = [max(0,min(200,x)) for x in v_next_noise]
            v_next_world_noise = list(self.env.ego2world(Point(v_next_noise)).coords)[0]
            # 
            self.env.update_historical_map(self.location,self.v_current_world,v_next_world_noise)
            if whether_sample and not end_v_flag:
                self.env.train_samples[self.location].append({'vt':v_current,'v_nexts':[list(expert_next.coords)[0]],'feature_tensor':feature_tensor,'segment_ahead':[self.segment_ahead]})
            if self.v_current_world == v_next_world_noise:
                self.cease = True
            self.v_current_world = v_next_world_noise
       
def sampler(args):
    
    # ===================
    args.sample_dir = f'./{args.savedir}/samples_detr'
    create_directory(f'./{args.savedir}/samples_detr_vis/BC',delete=True)
    create_directory(f'./{args.savedir}/samples_detr_train/BC',delete=True)

    data_conf = {
        'num_channels': 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
    canvas_h = int(patch_h / data_conf['ybound'][2])
    canvas_w = int(patch_w / data_conf['xbound'][2])
    patch_size = (patch_h, patch_w)
    canvas_size = (canvas_h, canvas_w)
    args.bsz = 1
    train_loader, nusc = centerlinedet_sampler_dataset(args, data_conf, train_set_selection='train2')

    #
    frame_index = 0
    pre_scene_name = None
    env = Environment(args,canvas_size,patch_size)
    with tqdm(total=len(train_loader), unit='img') as pbar:
        for i, data in enumerate(train_loader):
            # ground truth segmentation mask of the frame
            segment_mask = data['segment_mask'].unsqueeze(1).cuda()
            scene_name = data['name'][0]
            if scene_name!=pre_scene_name :
                pre_scene_name = scene_name
                frame_index = 0
            
            # load the predicted BEV segmentation map fusing neighboring frames
            pred_fused_segmentation = np.array(Image.open(f'./{args.savedir}/fused_segmentation/pred_mask/{scene_name}_{frame_index}.png'))
            initial_candidate_mask = np.array(Image.open(f'./{args.savedir}/fused_segmentation/pred_initial_candidate/{scene_name}_{frame_index}.png'))
            frame_index += 1
            
            if frame_index%args.sample_rate==0:
                # generate training samples on this frame
                if len(pred_fused_segmentation.shape)==3:
                    pred_fused_segmentation = pred_fused_segmentation[:,:,0]
                feature_tensor_raw = torch.Tensor(pred_fused_segmentation/255.0).unsqueeze(0).unsqueeze(0).cuda()
                initial_candidate_mask = torch.Tensor(initial_candidate_mask/255.0).unsqueeze(0).unsqueeze(0).cuda()
                env.update_pred_mask(torch.cat([feature_tensor_raw,initial_candidate_mask,segment_mask],dim=1))
                location,patch_box,patch_angle,rec,idx = data['info']
                env.generate_samples(location[0],[x[0].cpu().detach().numpy() for x in patch_box],patch_angle[0].cpu().detach().numpy(),rec)
            pbar.update()


if __name__ == '__main__':
    args = get_args()
    setup_seed(20)
    sampler(args)
