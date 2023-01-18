import os
import numpy as np
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import measure
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Polygon, box, Point
import json
import pickle
import cv2

import shutil
import sys
sys.path.append('..')
from config.get_args import get_args
from data.dataset import inference_centerlinedet_dataset
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

class Vertex():
    def __init__(self,v,id):
        self.x = v[0]
        self.y = v[1]
        self.id = id
        self.neighbors = []

class Edge():
    def __init__(self,src,dst,id):
        self.src = src
        self.dst = dst
        self.id = id

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertex_num = 0
        self.edge_num = 0

    def find_v(self,v_coord):
        if f'{v_coord[0]}_{v_coord[1]}' in self.vertices.keys():
            return self.vertices[f'{v_coord[0]}_{v_coord[1]}']
        return 

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.find_v(v1_coord)
        if v1 is None:
            v1 = Vertex(v1_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v1.x}_{v1.y}'] = v1
        
        v2 = self.find_v(v2_coord)
        if v2 is None:
            v2 = Vertex(v2_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v2.x}_{v2.y}'] = v2

        if v1 not in v2.neighbors:
            v2.neighbors.append(v1)
        if v2 not in v1.neighbors:
            v1.neighbors.append(v2)
        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1


class CenterLineInstance():
    def __init__(self, agent, src):
        self.agent = agent
        self.id = agent.instance_counter
        agent.instance_counter += 1
        self.first_vertex = src
        self.remove_first_vertex = False
        self.reverse_direction_add = False
        
        self.last_vertex = src
        self.remove_last_vertex = False
        self.direction_add = False

        self.linestring = None

    def whether_inside_patch(self,patch):
        if self.first_vertex.within(patch):
            self.reverse_direction_add = True
        else:
            self.reverse_direction_add = False
        if self.last_vertex.within(patch):
            self.direction_add = True
        else:
            self.direction_add = False

    def add_v(self,new_v):
        new_v = self.agent.frame2world(Point(new_v))
        if self.linestring is None:
            self.linestring = LineString(list(self.first_vertex.coords)+list(new_v.coords))
        else:
            self.linestring = LineString(list(self.linestring.coords)+list(new_v.coords))
        self.last_vertex = Point(list(self.linestring.coords)[-1])
        
    
    def add_reverse_v(self,new_v):
        new_v = self.agent.frame2world(Point(new_v))
        if self.linestring is None:
            self.linestring = LineString(list(new_v.coords) + list(self.first_vertex.coords))
        else:
            self.linestring = LineString(list(new_v.coords)+list(self.linestring.coords))
        self.first_vertex = Point(list(self.linestring.coords)[0])

COLORS = {
    # static
    'gt_centerline':         (255, 100, 100),
    'centerline':         (90, 90, 90),
    'inti_point':           (0, 200, 200),
    'box':                  (153,153,255),

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (130, 130, 130),

    # dynamic
    'car':                  (255, 158, 0),
    'truck':                (255, 99, 71),
    'bus':                  (255, 127, 80),
    'trailer':              (255, 140, 0),
    'construction':         (233, 150, 150),
    'lane':                 (110, 110, 110),
    'motorcycle':           (255, 61, 99),
    'bicycle':              (220, 20, 60),

    'nothing':              (200, 200, 200)
}


    

class CenterLineDetAgent():
    def __init__(self,args,net,scene_name):
        # args
        self.args = args
        self.net = net
        self.scene_name = scene_name
        # masks
        self.pred_segment_mask = None
        self.initial_candidate_mask = None
        self.segment_mask = None
        # transformation
        self.rotation = None
        self.translation = None
        self.scale = None
        # ROI patch
        self.patch = None
        self.frame_historical_map = Image.fromarray(np.zeros((200,200)))
        self.frame_initial_candidate = []
        self.frame_previous_frame = []
        self.frame_instances = []
        self.removed_frame_instance = []
        self.frame_index = 0
        # world
        self.world_historical_lines = []
        self.world_initial_candidates = []
        self.world_point_previous_frame = []
        self.world_instances = []
        self.world_past_points = []
        
        self.gt_points = []

        # counter 
        self.sample_counter = 0
        self.step_counter = 0
        self.instance_counter = 0

        # vis save, only for visualization
        with open(f'{self.args.dataroot}/label_world_crop/{self.scene_name}.json','r') as jf:
            info = json.load(jf)
        x_min, x_max, y_min, y_max = info['x_min'], info['x_max'], info['y_min'], info['y_max']
        self.vis_image = Image.fromarray(np.ones((y_max-y_min+1,x_max-x_min+1))*255).convert('RGB')
        self.vis_draw = ImageDraw.Draw(self.vis_image)
        self.ego_points = []
        self.scene_vis_info = {'seg':[],'history':[],\
                'init_new_world':[[]],'init_new_frame':[[]],\
                'init_new_world_valid':[[]],'init_new_frame_valid':[],\
                'init_frame_world_valid':[[]],'init_frame_frame_valid':[[]],\
                'steps_world':[[]],'steps_frame':[[]]}
        
    def update_frame(self,pred_segment_mask,initial_candidate_mask,segment_mask,canvas_size,patch_box,patch_angle,frame_index):
        '''
        Update info of the current frame

        :param pred_segment_mask: predicted BEV segmentation mask fusing neighboring frames
        :param initial_candidate_mask: candidate initial vertices segmentation map
        :param segment_mask: gt BEV segmentation mask
        :param canvas_size, patch_box, patch_angle: patch ROI info
        '''
        # patch geometry
        patch_x, patch_y, patch_h, patch_w = patch_box
        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0
        # info update
        self.pred_segment_mask = pred_segment_mask
        self.initial_candidate_mask = initial_candidate_mask
        self.scene_vis_info['seg'].append([self.pred_segment_mask.cpu().detach().numpy().tolist(),self.initial_candidate_mask.tolist()])
        self.segment_mask = segment_mask
        # self.step_counter = 0
        self.rotation = {'angle':patch_angle,'x':patch_x,'y':patch_y}
        self.translation = {'x':trans_x,'y':trans_y}
        self.scale = {'width':scale_width,'height':scale_height}
        self.patch = self.get_patch_coord([patch_x, patch_y, patch_h, patch_w],patch_angle)
        self.frame_historical_map = Image.fromarray(np.zeros((200,200)))
        self.sample_counter += 1
        self.frame_index = frame_index - 1
        # ROI patch
        self.frame_instances = []
        self.removed_frame_instance = []
        # frame candidate initial vertices extraction
        self.extract_initial_candidates()
    
    def extract_initial_candidates(self):
        '''
        Extract candidate initial vertices from the current frame
        '''
        # initial_candidate_mask = self.initial_candidate_mask / np.max(self.initial_candidate_mask)
        labels = measure.label(self.initial_candidate_mask, connectivity=2)
        props = measure.regionprops(labels)
        max_area = self.args.area_thr
        for region in props:
            if region.area > max_area:
                center_point = Point(region.centroid[::-1])
                self.scene_vis_info['init_new_frame'][-1].append(region.centroid[::-1])
                Center_world = list(self.frame2world(center_point).coords)[0]
                self.world_initial_candidates.insert(0,Center_world)
                self.scene_vis_info['init_new_world'][-1].append(Center_world)
        self.scene_vis_info['init_new_world'].append([])
        self.scene_vis_info['init_new_frame'].append([])

    def crop_tensor(self,v_current):
        '''
        Crop ROI centering on v_current
        :param v_current: current vertex coordinates
        :return cropped_tensor: cropped ROI centering on v_current. 2 channels, 1st channel is segmentation,
            2nd channel is historical map.
        '''
        history_map = torch.Tensor(np.array(self.frame_historical_map)).unsqueeze(0)/255
        feature_tensor = torch.cat([self.pred_segment_mask,self.segment_mask.unsqueeze(0)],dim=1)
        pad_feature_tensor = F.pad(feature_tensor[0],(self.args.ROI_half_length+20,self.args.ROI_half_length+20,self.args.ROI_half_length+20,self.args.ROI_half_length+20,0,0),'constant',0).cpu()
        pad_history_map =  F.pad(history_map,(self.args.ROI_half_length+20,self.args.ROI_half_length+20,self.args.ROI_half_length+20,self.args.ROI_half_length+20,0,0),'constant',0)
        cat_tensor = torch.cat([pad_feature_tensor,pad_history_map],dim=0)
        cropped_tensor = cat_tensor[:,int(v_current[1]+20):int(v_current[1]+20+self.args.ROI_half_length*2),int(v_current[0]+20):int(v_current[0]+20+self.args.ROI_half_length*2)]
        return cropped_tensor.unsqueeze(0).cuda()
    
    def extract_valid_coords(self,v_current,v_previous,pred_logits,pred_coord,thr=0.5):
        '''
        Extract valid coordinates from predictions.

        :param v_current: current vertex
        :param pred_logits: logits of predicted coordinates
        :param pred_coords: predicted vertex coordinates (translation of v_current)
        :param thr: threshold for pred_logits
        :return output: extracted valid coords
        '''
        pred_coord = pred_coord[0]
        pred_coord = pred_coord.cpu().detach().numpy().tolist()
        pred_coord = [[x[0]*self.args.ROI_half_length+v_current[0],x[1]*self.args.ROI_half_length+v_current[1]] for x in pred_coord]
        pred_coord_all = pred_coord.copy()
        pred_logits = pred_logits[0].softmax(dim=1)
        valid_vertices = []
        for ii, coord in enumerate(pred_coord):
            if pred_logits[ii,0] >= thr and coord[0]>=0 and coord[0]<200 and coord[1]>=0 and coord[1]<200:
                valid_vertices.append(coord)
        # filter by angle
        vector_previous = np.array(v_current) - np.array(v_previous)
        norm_previous = np.linalg.norm(vector_previous)
        filtered_vertices = []
        filtered_vectors = []
        output = []
        for v in valid_vertices:
            save_flag = True
            vector_v = np.array(v) - np.array(v_current)
            norm_v = np.linalg.norm(vector_v)
            if not norm_previous or (vector_v.dot(vector_previous/norm_previous)>0 and norm_v):
                if norm_v>1:
                    vector_v = vector_v / norm_v
                    for i, (norm_u,vector_u) in enumerate(filtered_vectors):
                        if vector_v.dot(vector_u)>0.99 and norm_u and norm_v:
                            if norm_v>norm_u and v in output:
                                output.remove(v)
                            elif norm_v<norm_u:
                                save_flag = False
                    if save_flag:
                        filtered_vertices.append(v)
                        filtered_vectors.append([norm_v,vector_v])
                        output.append(v)
        return output, pred_coord_all

    def get_patch_coord(self,patch_box,patch_angle) -> Polygon:
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
    
    def frame2world(self,shapely_object):
        '''
        Transform the input object from frame coordinate to world coordinate

        :param shapely object: object in frame coordinate
        :return shapely object: object in world coordinate
        '''
        shapely_object = affinity.scale(shapely_object, xfact=1/self.scale['width'], yfact=1/self.scale['height'], origin=(0, 0))
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, -self.translation['x'], -self.translation['y']])
        shapely_object = affinity.rotate(shapely_object, self.rotation['angle'], origin=(self.rotation['x'],self.rotation['y']), use_radians=False)
        return shapely_object
    
    def world2frame(self,shapely_object):
        '''
        Transform the input object from world coordinate to frame coordinate

        :param shapely object: object in world coordinate
        :return shapely object: object in frame coordinate
        '''
        shapely_object = affinity.rotate(shapely_object, -self.rotation['angle'], origin=(self.rotation['x'],self.rotation['y']), use_radians=False)
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, self.translation['x'], self.translation['y']])
        shapely_object = affinity.scale(shapely_object, xfact=self.scale['width'], yfact=self.scale['height'], origin=(0, 0))
        return shapely_object

    def world_objects_to_local(self):
        '''
        Project world instances to local frame
        '''
        # update endpoints in the current frame of world instance
        for instance in self.world_instances:
            instance.whether_inside_patch(self.patch)

        # draw historical lines (generated in past frames) in the current frame historical map
        draw = ImageDraw.Draw(self.frame_historical_map)
        for line in self.world_historical_lines:
            inter_line = line.intersection(self.patch)
            inter_line = self.world2frame(inter_line)
            if not inter_line.is_empty and not inter_line.boundary.is_empty:
                if inter_line.geom_type == 'MultiLineString':
                    for inter_line_single in inter_line:
                        p1, p2 = inter_line_single.boundary
                        draw.line([tuple(p1.coords)[0],tuple(p2.coords)[0]],fill='white',width=1)
                else:
                    p1, p2 = inter_line.boundary
                    draw.line([tuple(p1.coords)[0],tuple(p2.coords)[0]],fill='white',width=1)
        # filter initial candidates in the current frame (those candidates that are too closed to 
        # the foreground of the historical map)
        frame_initial_candidates = []
        for world_initical_candidate in self.world_initial_candidates:
            frame_initial_candidates.append(list(self.world2frame(Point(world_initical_candidate)).coords)[0])
        self.world_initial_candidates = []
        if len(frame_initial_candidates):
            history_map_pixels = np.where(np.array(self.frame_historical_map)!=0)
            history_map_pixels = [[history_map_pixels[1][x],history_map_pixels[0][x]] for x in range(len(history_map_pixels[0]))]
            if len(history_map_pixels): 
                tree = cKDTree(history_map_pixels)
                dds,_ = tree.query(frame_initial_candidates,k=1)
                frame_initial_candidates = [frame_initial_candidates[j] for j in range(len(dds)) if dds[j]>self.args.filter_distance]
        
        # add new instances from new initial candidates
        for frame_initial_candidate in frame_initial_candidates:
            new_instance = CenterLineInstance(self,self.frame2world(Point(frame_initial_candidate)))
            self.world_instances.append(new_instance)
            new_instance.direction_add = True
        
        # filter by removing vertices that are too closed to other vertices
        for ii, ins1 in enumerate(self.world_instances[:-1]):
            if ins1.remove_first_vertex and ins1.remove_last_vertex:
                continue
            for jj, ins2 in enumerate(self.world_instances[ii+1:]):
                if not ins2.remove_last_vertex and \
                    ((not ins1.remove_last_vertex and np.linalg.norm(np.array(list(ins1.last_vertex.coords)[0])-np.array(list(ins2.last_vertex.coords)[0]))<self.args.world_alignment_distance)\
                    or (not ins1.remove_first_vertex and np.linalg.norm(np.array(list(ins1.first_vertex.coords)[0])-np.array(list(ins2.last_vertex.coords)[0]))<self.args.world_alignment_distance)):
                    ins2.remove_last_vertex = True
                if not ins2.remove_first_vertex and \
                    ((not ins1.remove_last_vertex and np.linalg.norm(np.array(list(ins1.last_vertex.coords)[0])-np.array(list(ins2.first_vertex.coords)[0]))<self.args.world_alignment_distance)\
                    or (not ins1.remove_first_vertex and np.linalg.norm(np.array(list(ins1.first_vertex.coords)[0])-np.array(list(ins2.first_vertex.coords)[0]))<self.args.world_alignment_distance)):
                    ins2.remove_first_vertex = True

        # find initial candidates in the current frame
        for ii, instance in enumerate(self.world_instances):
            if instance.direction_add or instance.reverse_direction_add:
                self.frame_instances.append(instance)
        self.removed_frame_instance = []
        
    def add_line(self,frame_v1,frame_v2):
        '''
        Add a LineString to the histoical buffer in world coordinate system
        :param frame_v1: src
        :param frame_v2: dst
        '''
        world_v1 = list(self.frame2world(Point(frame_v1)).coords)[0]
        world_v2 = list(self.frame2world(Point(frame_v2)).coords)[0]
        self.world_historical_lines.append(LineString([tuple(world_v1),tuple(world_v2)]))

    def alignment(self,instance_current,v_next):
        '''
        If one predicted vertex is closed enough to an endpoint of an CenterLineDet instance,
        move the predicted vertex to that endpoints, and stop the current instance.

        :param instance_current: CenterLineDet instance that the model currently works on 
        :param v_next: predicted vertex in the next step
        :return stop_action (bool): if closed enough endpoint detected, stop the current instance
        :return v_next: coordinates that the predicted v_next moves to
        '''
        for instance in self.removed_frame_instance + self.frame_instances:
            if  instance!=instance_current:
                v_next_world = list(self.frame2world(Point(v_next)).coords)[0]
                first_vertex = list(instance.first_vertex.coords)[0]
                if np.linalg.norm(np.array(v_next_world)-np.array(first_vertex)) <self.args.world_alignment_distance:
                    v_next = list(self.world2frame(Point(first_vertex)).coords)[0]
                    return True, v_next
                
                last_vertex = list(instance.last_vertex.coords)[0]
                if np.linalg.norm(np.array(v_next_world)-np.array(last_vertex)) <self.args.world_alignment_distance:
                    v_next = list(self.world2frame(Point(last_vertex)).coords)[0]
                    return True, v_next
        return False, v_next
    
    def vis_pred(self,image,info=None,init_point=False):
        im = np.zeros((image.shape[0],image.shape[1],3))
        im[:,:,0] = image
        im[:,:,1] = image
        im[:,:,2] = image
        im = im.astype(np.float64)
        kernel = np.ones((3,3), np.uint8)
        im = cv2.dilate(im, kernel)
        im *= np.array(COLORS['centerline']).astype(np.uint8)
        im = Image.fromarray(im.astype(np.uint8))
        im = ImageOps.invert(im)
        return im
    
    def draw_points(self,x,y,draw,pred_coords_ROI_valid,pred_coords_ROI_all,v_current):

        draw.ellipse([x-3,y-3,x+3,y+3],fill=(255,128,185),outline=(255,128,185))
        
        if len(pred_coords_ROI_all):
            for jj in range(len(pred_coords_ROI_all)):
                v = pred_coords_ROI_all[jj]
                v = [x+v[0]-v_current[0],y+v[1]-v_current[1]]
                draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(30,144,255))

        if pred_coords_ROI_valid:
            for jj in range(len(pred_coords_ROI_valid)):
                v = pred_coords_ROI_valid[jj]
                v = [x+v[0]-v_current[0],y+v[1]-v_current[1]]
                draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(124,252,0))
    
    def draw_world_points(self,x,y,draw,pred_coords_ROI_valid,pred_coords_ROI_all,info,v_current):
        
        wv = self.convert(v_current,info)
        draw.ellipse([wv[0]+x-3,wv[1]+y-3,wv[0]+x+3,wv[1]+y+3],fill=(255,128,185),outline=(255,128,185))
        if len(pred_coords_ROI_all):
            for jj in range(len(pred_coords_ROI_all)):
                v = pred_coords_ROI_all[jj]
                wv = self.convert(v,info)
                draw.ellipse((wv[0]+x-2,wv[1]+y-2,wv[0]+x+2,wv[1]+y+2),fill=(30,144,255))

        if pred_coords_ROI_valid:
            for jj in range(len(pred_coords_ROI_valid)):
                v = pred_coords_ROI_valid[jj]
                wv = self.convert(v,info)
                draw.ellipse((wv[0]+x-2,wv[1]+y-2,wv[0]+x+2,wv[1]+y+2),fill=(124,252,0))

    def draw_past_vertices(self,draw,vertices):
        for v in vertices:
            draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,215,0),outline=(255,215,0))
    
    def convert(self,v,info):
        x_min, x_max, y_min, y_max = info['x_min'], info['x_max'], info['y_min'], info['y_max']
        v = list(self.frame2world(Point(v)).coords[0])
        v[0] = v[0]*4 - x_min
        v[1] = v[1]*4 - y_min
        return v
    
    def step(self):
        '''
        Step function of the agent
        '''
        self.world_objects_to_local()
        draw_history_map = ImageDraw.Draw(self.frame_historical_map)
        self.frame_historical_map.convert('RGB').save(os.path.join(self.args.save_dir_single,'pred_mask',f'{self.scene_name}_{self.sample_counter-1}.png'))
        # vis
        with open(f'{self.args.dataroot}/label_world_crop/{self.scene_name}.json','r') as jf:
            info = json.load(jf)
        x_min, x_max, y_min, y_max = info['x_min'], info['x_max'], info['y_min'], info['y_max']
        
        height, width = y_max-y_min+1,x_max-x_min+1

        gt_mask = Image.open('../visualization/seg_video/vis/gt_scene-0054.png')
        ego_gt_mask = Image.open(f'../visualization/seg_video/vis_seg_single/gt_scene-0054_{self.frame_index}.png')
        ego_pred_mask = Image.open(f'../visualization/seg_video/vis_seg_single/pred_scene-0054_{self.frame_index}.png')
        
        while len(self.frame_instances)and self.step_counter<200:
            if len(self.frame_instances):
                instance = self.frame_instances.pop(0)
                self.removed_frame_instance.append(instance)
            else:
                break
            # iterations
            # add from last vertex of the instance
            feature_tensor = None
            if instance.direction_add and not instance.remove_last_vertex:
                v_current = [x for x in list(self.world2frame(instance.last_vertex).coords)[0]]
                v_current_raw = v_current.copy()
                v_previous = v_current
                instance_counter = 0
                self.scene_vis_info['steps_frame'][-1].append([[v_current]])
                self.scene_vis_info['steps_world'][-1].append([[list(instance.last_vertex.coords)[0]]])
                # iterations
                while instance_counter < 40:
                    end_flag = False
                    instance_counter += 1
                    self.step_counter += 1
                    # Process feature tensor      
                    feature_tensor = self.crop_tensor(v_current)
                    # predict vertices in the next step
                    action_pred = self.net(feature_tensor[:,[0,2]])
                    pred_coords, pred_coords_all = self.extract_valid_coords(v_current,v_previous,action_pred['pred_logits'],action_pred['pred_boxes'],thr=self.args.logit_thr)
                    v_previous = v_current
                    if len(pred_coords):
                        self.scene_vis_info['steps_frame'][-1][-1].append(pred_coords)
                        self.scene_vis_info['steps_world'][-1][-1].append([list(self.frame2world(Point(v)).coords)[0] for v in pred_coords])
                    # take corresponding actions based on the number of valid predicted vertices
                    if not len(pred_coords):
                        end_flag = True
                    elif len(pred_coords) == 1:
                        draw_history_map.line([v_current[0],v_current[1],pred_coords[0][0],pred_coords[0][1]],fill='white',width=1)
                        stop_action, pred_coords[0] = self.alignment(instance,pred_coords[0])
                        self.add_line(v_current,pred_coords[0])
                        v_current = pred_coords[0]
                        instance.add_v(v_current)
                        if stop_action:
                            instance.remove_first_vertex = True
                            end_flag = True
                        if instance_counter>=40 or v_current[0]>195 or v_current[1]>195 or v_current[1]<5 :
                            end_flag = True
                    else:
                        instance.remove_last_vertex = True
                        for idx in range(len(pred_coords)):
                            stop_action, pred_coords[0] = self.alignment(instance,pred_coords[idx])
                            self.add_line(v_current,pred_coords[idx])
                            draw_history_map.line([v_current[0],v_current[1],pred_coords[idx][0],pred_coords[idx][1]],fill='white',width=1)
                            # add new instance 
                            if not stop_action:
                                new_instance = CenterLineInstance(self,self.frame2world(Point(v_current)))
                                new_instance.add_v(pred_coords[idx])
                                new_instance.remove_first_vertex = True
                                self.world_instances.append(new_instance)
                                self.frame_instances.append(new_instance)
                        end_flag = True
                    
                    dst = Image.new('RGB',(width*2+200+20*3+40*2+40+40+100+150+100,height+150+40),(200,200,200))
                    draw = ImageDraw.Draw(dst)
                    draw.rectangle([0,0,width*2+60*2,150+40+height],fill=(255,229,204))
                    draw.rectangle([width*2+60*2,0,width*2+60*2+100+300,150+40+height-100],fill=(240,248,255))
                    draw.rectangle([width*2+60*2+100+300,0,width*2+60*2+100+300+100+150,150+40+height-100],fill=(255,240,240))
                    dst.paste(gt_mask,(40,150))
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 40)
                    draw.text((40+width//2-120, 80+20),"Ground Truth",(150,150,150),font=font)

                    # dst.paste(pred_mask,(40+20+width,150))
                    draw.text((40+20+width+width//2-150, 80+20),"Fused Prediction",(150,150,150),font=font)
                    # Ego
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 30)
                    draw.text((width*2+60*2+100, 140),"Ground Truth",(100,100,100),font=font)
                    dst.paste(ego_gt_mask,(width*2+60*2+100,190))

                    draw.text((width*2+60*2+130, 500-50),"Prediction",(100,100,100),font=font)
                    dst.paste(ego_pred_mask,(width*2+60*2+100,150 + 200 + 150))
                    #
                    for u in self.scene_vis_info['init_new_frame_valid']:
                        v = [width*2+60*2+100+u[0],190+u[1]]
                        draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,0,0))
                        v = [width*2+60*2+100+u[0],150 + 200 + 150+u[1]]
                        draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,0,0))
                    #
                    top, bottom, left, right = v_current_raw[1]-75, v_current_raw[1]+75, v_current_raw[0]-75,v_current_raw[0]+75
                    
                    # Agent
                    ROI = ego_pred_mask.crop((left, top, right, bottom))
                    ROI = np.array(ROI)
                    ROI[np.where(ROI[:,:,0]==0)]=(255,255,255)
                    ROI = Image.fromarray(ROI.astype(np.uint8))
                    his = self.vis_pred(feature_tensor[0,-1].cpu().detach().numpy() * 255)
                    draw.text((width*2+60*2+50+100+350, 140),"ROI",(100,100,100),font=font)
                    dst.paste(ROI,(width*2+60*2+50+100+300,190))

                    draw.text((width*2+60*2+80+100+280, 500-50),"Historical",(100,100,100),font=font)
                    dst.paste(his,(width*2+60*2+50+100+300,150 + 200 + 150))
                    
                    # 
                    self.draw_points(width*2+60*2+100+left+75,150 + 200 + 150+top+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    self.draw_points(width*2+60*2+100+left+75,190+top+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    
                    self.draw_points(width*2+60*2+50+100+300+75,190+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    self.draw_points(width*2+60*2+50+100+300+75,150 + 200 + 150+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    
                    v = [width*2+60*2+100+left+75,150 + 200 + 150+top+75]
                    draw.ellipse((v[0]-3,v[1]-3,v[0]+3,v[1]+3),fill=(255,128,185))
                    v = [width*2+60*2+100+left+75,190+top+75]
                    draw.ellipse((v[0]-3,v[1]-3,v[0]+3,v[1]+3),fill=(255,128,185))
                    #
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 50)
                    draw.text((width*2+60*2+250, 150+40+height-70),f"{self.frame_index}-th frame",(100,100,100),font=font)
                    #
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 50)
                    draw.text((40+20+width-70, 20),"World",(255,153,51),font=font)
                    draw.text((width*2+60*2+150, 20),"Ego",(176,196,222),font=font)
                    draw.text((width*2+60*2+100+60+300, 20),"Agent",(255,128,185),font=font)
                    
                    # ego
                    left = max(0,min(left,200))
                    right = max(0,min(right,200))
                    top = max(0,min(top,200))
                    bottom = max(0,min(bottom,200))
                    draw.rectangle([width*2+60*2+100+left,150 + 200 + 150+top,width*2+60*2+100+right,150 + 200 + 150+bottom],fill=None,outline=(255,128,185),width=3)
                    draw.rectangle([width*2+60*2+100+left,190+top,width*2+60*2+100+right,190+bottom],fill=None,outline=(255,128,185),width=3)
                    
                    # world
                    wc = self.convert(v_current_raw,info)
                    self.ego_points.append(wc)
                    if len(pred_coords)==1:
                        v = pred_coords[0]
                        wv = self.convert(v,info)
                        
                        self.ego_points.append(wv)
                        self.vis_draw.line([wv[0],wv[1],wc[0],wc[1]],fill=(255,128,185),width=4)
                    elif len(pred_coords)>1:
                        for pi in range(len(pred_coords)):
                            v = pred_coords[pi]
                            wv = self.convert(v,info)
                            self.ego_points.append(wv)
                            self.vis_draw.line([wv[0],wv[1],wc[0],wc[1]],fill=(255,128,185),width=4)
                    for v in self.ego_points:
                        self.vis_draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,215,0),outline=(255,215,0))
                    #
                    self.draw_past_vertices(self.vis_draw,self.ego_points)
                    dst.paste(self.vis_image,(40+width+20,150))

                    self.draw_world_points(40+width+20,150,draw,pred_coords, pred_coords_all,info,v_current_raw)
                    
                    a,b,c,d = [0,0],[200,0],[200,200],[0,200]
                    wa,wb,wc,wd = self.convert(a,info),self.convert(b,info),self.convert(c,info),self.convert(d,info)
                    draw.line([wa[0]+40+width+20,wa[1]+150,wb[0]+40+width+20,wb[1]+150],fill=(176,196,222),width=3)
                    draw.line([wb[0]+40+width+20,wb[1]+150,wc[0]+40+width+20,wc[1]+150],fill=(176,196,222),width=3)
                    draw.line([wc[0]+40+width+20,wc[1]+150,wd[0]+40+width+20,wd[1]+150],fill=(176,196,222),width=3)
                    draw.line([wd[0]+40+width+20,wd[1]+150,wa[0]+40+width+20,wa[1]+150],fill=(176,196,222),width=3)

                    a,b,c,d = [v_current_raw[0]-75,v_current_raw[1]-75],[v_current_raw[0]-75,v_current_raw[1]+75],[v_current_raw[0]+75,v_current_raw[1]+75],[v_current_raw[0]+75,v_current_raw[1]-75]
                    wa,wb,wc,wd = self.convert(a,info),self.convert(b,info),self.convert(c,info),self.convert(d,info)
                    draw.line([wa[0]+40+width+20,wa[1]+150,wb[0]+40+width+20,wb[1]+150],fill=(255,128,185),width=3)
                    draw.line([wb[0]+40+width+20,wb[1]+150,wc[0]+40+width+20,wc[1]+150],fill=(255,128,185),width=3)
                    draw.line([wc[0]+40+width+20,wc[1]+150,wd[0]+40+width+20,wd[1]+150],fill=(255,128,185),width=3)
                    draw.line([wd[0]+40+width+20,wd[1]+150,wa[0]+40+width+20,wa[1]+150],fill=(255,128,185),width=3)

                    
                    dst.save(f'{args.save_dir_multi}/seg_video_frame/{self.step_counter:09d}.png')
                    v_current_raw = list(v_current).copy()
                    if end_flag is True:
                        break
            # add from first vertex of the instance
            if instance.reverse_direction_add and not instance.remove_first_vertex:
                v_current = [(x) for x in list(self.world2frame(instance.first_vertex).coords)[0]]
                v_current_raw = v_current.copy()
                v_previous = v_current
                instance_counter = 0
                self.scene_vis_info['steps_frame'][-1].append([[v_current]])
                self.scene_vis_info['steps_world'][-1].append([[list(instance.first_vertex.coords)]])
                # iterations
                while instance_counter < 40:
                    end_flag = False
                    instance_counter += 1
                    self.step_counter += 1
                    # Process feature tensor      
                    feature_tensor = self.crop_tensor(v_current)
                    # predict vertices in the next step
                    action_pred = self.net(feature_tensor[:,[0,2]])
                    pred_coords, pred_coords_all = self.extract_valid_coords(v_current,v_previous,action_pred['pred_logits'],action_pred['pred_boxes'],thr=self.args.logit_thr)
                    v_previous = v_current
                    if len(pred_coords):
                        self.scene_vis_info['steps_frame'][-1][-1].append(pred_coords)
                        self.scene_vis_info['steps_world'][-1][-1].append([list(self.frame2world(Point(v)).coords)[0] for v in pred_coords])
                    # take corresponding actions based on the number of valid predicted vertices
                    if not len(pred_coords):
                        end_flag = True
                    elif len(pred_coords) == 1:
                        stop_action, pred_coords[0] = self.alignment(instance,pred_coords[0])
                        draw_history_map.line([v_current[0],v_current[1],pred_coords[0][0],pred_coords[0][1]],fill='white',width=1)
                        self.add_line(v_current,pred_coords[0])
                        v_current = pred_coords[0]
                        instance.add_reverse_v(v_current)
                        if stop_action:
                            instance.remove_last_vertex = True
                            end_flag = True
                        if instance_counter>=40 or v_current[0]>195 or v_current[1]>195 or v_current[1]<5 :
                            end_flag = True
                    else:
                        instance.remove_first_vertex = True
                        for idx in range(len(pred_coords)):
                            stop_action, pred_coords[0] = self.alignment(instance,pred_coords[idx])
                            self.add_line(v_current,pred_coords[idx])
                            draw_history_map.line([v_current[0],v_current[1],pred_coords[idx][0],pred_coords[idx][1]],fill='white',width=1)
                            # add new instance 
                            if not stop_action:
                                new_instance = CenterLineInstance(self,self.frame2world(Point(v_current)))
                                new_instance.add_v(pred_coords[idx])
                                new_instance.remove_first_vertex = True
                                self.world_instances.append(new_instance)
                                self.frame_instances.append(new_instance)
                        end_flag = True
                    
                    dst = Image.new('RGB',(width*2+200+20*3+40*2+40+40+100+150+100,height+150+40),(200,200,200))
                    draw = ImageDraw.Draw(dst)
                    draw.rectangle([0,0,width*2+60*2,150+40+height],fill=(255,229,204))
                    draw.rectangle([width*2+60*2,0,width*2+60*2+100+300,150+40+height-100],fill=(240,248,255))
                    draw.rectangle([width*2+60*2+100+300,0,width*2+60*2+100+300+100+150,150+40+height-100],fill=(255,240,240))
                    dst.paste(gt_mask,(40,150))
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 40)
                    draw.text((40+width//2-120, 80+20),"Ground Truth",(150,150,150),font=font)

                    # dst.paste(pred_mask,(40+20+width,150))
                    draw.text((40+20+width+width//2-150, 80+20),"Fused Prediction",(150,150,150),font=font)
                    # Ego
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 30)
                    draw.text((width*2+60*2+100, 140),"Ground Truth",(100,100,100),font=font)
                    dst.paste(ego_gt_mask,(width*2+60*2+100,190))

                    draw.text((width*2+60*2+130, 500-50),"Prediction",(100,100,100),font=font)
                    dst.paste(ego_pred_mask,(width*2+60*2+100,150 + 200 + 150))
                    #
                    for u in self.scene_vis_info['init_new_frame_valid']:
                        v = [width*2+60*2+100+u[0],190+u[1]]
                        draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,0,0))
                        v = [width*2+60*2+100+u[0],150 + 200 + 150+u[1]]
                        draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,0,0))
                    #
                    top, bottom, left, right = v_current_raw[1]-75, v_current_raw[1]+75, v_current_raw[0]-75,v_current_raw[0]+75
                    
                    # Agent
                    ROI = ego_pred_mask.crop((left, top, right, bottom))
                    ROI = np.array(ROI)
                    ROI[np.where(ROI[:,:,0]==0)]=(255,255,255)
                    ROI = Image.fromarray(ROI.astype(np.uint8))
                    his = self.vis_pred(feature_tensor[0,-1].cpu().detach().numpy() * 255)
                    draw.text((width*2+60*2+50+100+350, 140),"ROI",(100,100,100),font=font)
                    dst.paste(ROI,(width*2+60*2+50+100+300,190))

                    draw.text((width*2+60*2+80+100+280, 500-50),"Historical",(100,100,100),font=font)
                    dst.paste(his,(width*2+60*2+50+100+300,150 + 200 + 150))
                    
                    # 
                    self.draw_points(width*2+60*2+100+left+75,150 + 200 + 150+top+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    self.draw_points(width*2+60*2+100+left+75,190+top+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    
                    self.draw_points(width*2+60*2+50+100+300+75,190+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    self.draw_points(width*2+60*2+50+100+300+75,150 + 200 + 150+75,draw,pred_coords, pred_coords_all, v_current_raw)
                    
                    v = [width*2+60*2+100+left+75,150 + 200 + 150+top+75]
                    draw.ellipse((v[0]-3,v[1]-3,v[0]+3,v[1]+3),fill=(255,128,185))
                    v = [width*2+60*2+100+left+75,190+top+75]
                    draw.ellipse((v[0]-3,v[1]-3,v[0]+3,v[1]+3),fill=(255,128,185))
                    #
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 50)
                    draw.text((width*2+60*2+250, 150+40+height-70),f"{self.frame_index}-th frame",(100,100,100),font=font)
                    #
                    font = ImageFont.truetype("../visualization/seg_video/Arial Bold.ttf", 50)
                    draw.text((40+20+width-70, 20),"World",(255,153,51),font=font)
                    draw.text((width*2+60*2+150, 20),"Ego",(176,196,222),font=font)
                    draw.text((width*2+60*2+100+60+300, 20),"Agent",(255,128,185),font=font)
                    
                    # ego
                    left = max(0,min(left,200))
                    right = max(0,min(right,200))
                    top = max(0,min(top,200))
                    bottom = max(0,min(bottom,200))
                    draw.rectangle([width*2+60*2+100+left,150 + 200 + 150+top,width*2+60*2+100+right,150 + 200 + 150+bottom],fill=None,outline=(255,128,185),width=3)
                    draw.rectangle([width*2+60*2+100+left,190+top,width*2+60*2+100+right,190+bottom],fill=None,outline=(255,128,185),width=3)
                    
                    # world
                    wc = self.convert(v_current_raw,info)
                    self.ego_points.append(wc)
                    if len(pred_coords)==1:
                        v = pred_coords[0]
                        wv = self.convert(v,info)
                        
                        self.ego_points.append(wv)
                        self.vis_draw.line([wv[0],wv[1],wc[0],wc[1]],fill=(255,128,185),width=4)
                    elif len(pred_coords)>1:
                        for pi in range(len(pred_coords)):
                            v = pred_coords[pi]
                            wv = self.convert(v,info)
                            self.ego_points.append(wv)
                            self.vis_draw.line([wv[0],wv[1],wc[0],wc[1]],fill=(255,128,185),width=4)
                    for v in self.ego_points:
                        self.vis_draw.ellipse((v[0]-2,v[1]-2,v[0]+2,v[1]+2),fill=(255,215,0),outline=(255,215,0))
                    #
                    self.draw_past_vertices(self.vis_draw,self.ego_points)
                    dst.paste(self.vis_image,(40+width+20,150))

                    self.draw_world_points(40+width+20,150,draw,pred_coords, pred_coords_all,info,v_current_raw)
                    
                    a,b,c,d = [0,0],[200,0],[200,200],[0,200]
                    wa,wb,wc,wd = self.convert(a,info),self.convert(b,info),self.convert(c,info),self.convert(d,info)
                    draw.line([wa[0]+40+width+20,wa[1]+150,wb[0]+40+width+20,wb[1]+150],fill=(176,196,222),width=3)
                    draw.line([wb[0]+40+width+20,wb[1]+150,wc[0]+40+width+20,wc[1]+150],fill=(176,196,222),width=3)
                    draw.line([wc[0]+40+width+20,wc[1]+150,wd[0]+40+width+20,wd[1]+150],fill=(176,196,222),width=3)
                    draw.line([wd[0]+40+width+20,wd[1]+150,wa[0]+40+width+20,wa[1]+150],fill=(176,196,222),width=3)

                    a,b,c,d = [v_current_raw[0]-75,v_current_raw[1]-75],[v_current_raw[0]-75,v_current_raw[1]+75],[v_current_raw[0]+75,v_current_raw[1]+75],[v_current_raw[0]+75,v_current_raw[1]-75]
                    wa,wb,wc,wd = self.convert(a,info),self.convert(b,info),self.convert(c,info),self.convert(d,info)
                    draw.line([wa[0]+40+width+20,wa[1]+150,wb[0]+40+width+20,wb[1]+150],fill=(255,128,185),width=3)
                    draw.line([wb[0]+40+width+20,wb[1]+150,wc[0]+40+width+20,wc[1]+150],fill=(255,128,185),width=3)
                    draw.line([wc[0]+40+width+20,wc[1]+150,wd[0]+40+width+20,wd[1]+150],fill=(255,128,185),width=3)
                    draw.line([wd[0]+40+width+20,wd[1]+150,wa[0]+40+width+20,wa[1]+150],fill=(255,128,185),width=3)

                    
                    dst.save(f'{args.save_dir_multi}/seg_video_frame/{self.step_counter:09d}.png')
                    v_current_raw = list(v_current).copy()

                    if end_flag is True:
                        break
    def vis(self,save_graph=False):
        '''
        Visualize predicted centerline graph in world and save
        '''
        with open(f'{self.args.dataroot}/label_world_crop/{self.scene_name}.json','r') as jf:
            info = json.load(jf)
        x_min, x_max, y_min, y_max = info['x_min'], info['x_max'], info['y_min'], info['y_max']
        world_image = Image.fromarray(np.zeros((y_max-y_min+1,x_max-x_min+1))).convert('RGB')
        draw = ImageDraw.Draw(world_image)
        output_edges = []
        for line in self.world_historical_lines:
            if line.boundary.is_empty: continue
            p1,p2 = line.boundary
            p1 = list(p1.coords)[0]
            p1 = [int(4*p1[0]-x_min),int(4*p1[1]-y_min)]
            p2 = list(p2.coords)[0]
            p2 = [int(4*p2[0]-x_min),int(4*p2[1]-y_min)]
            output_edges.append([p1,p2])
            draw.line([tuple(p1),tuple(p2)],fill='white',width=1)
        world_image = np.array(world_image.convert('RGB'))[:,:,0]
        # filter short segments
        labels = measure.label(world_image, connectivity=2)
        indexes = np.unique(labels)[1:]
        max_area = 100
        for index in indexes:
            if len(np.where(labels==index)[0]) < max_area:
                labels[np.where(labels==index)] = 0
        # save
        world_image = Image.fromarray(((labels!=0)*255).astype(np.uint8)).save(os.path.join(self.args.save_dir_multi,'pred_mask',f'{self.scene_name}.png'))
        
        
        with open(os.path.join(self.args.save_dir_multi,'pred_csv',f'{self.scene_name}.json'),'w') as jf:
            json.dump(self.scene_vis_info,jf)
            
        
def inference(args):

    # ============== 
    args.save_dir_single = f'./{args.savedir}/vis/single'
    create_directory(f'./{args.savedir}/vis/single/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/vis/single/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/vis/single/pred_init',delete=True)

    args.save_dir_multi = f'./{args.savedir}/vis/multi'
    create_directory(f'./{args.savedir}/vis/multi/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/vis/multi/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/vis/multi/pred_graph',delete=True)
    create_directory(f'./{args.savedir}/vis/multi/pred_csv',delete=True)
    create_directory(f'./{args.savedir}/vis/multi/seg_video_frame',delete=True)

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
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
    canvas_h = int(patch_h / data_conf['ybound'][2])
    canvas_w = int(patch_w / data_conf['xbound'][2])
    canvas_size = (canvas_h, canvas_w)

    val_loader = inference_centerlinedet_dataset(args, data_conf, vis=True)
    
    CenterLineDet, _ = build_model(args)
    CenterLineDet.load_state_dict(torch.load(args.centerlinedet_checkpoint_dir,map_location='cpu')['model'])
    CenterLineDet.cuda()
    CenterLineDet.eval()
    
    pre_scene = None
    agent = None
    frame_index = 0
    with torch.no_grad():
        with tqdm(total=len(val_loader), unit='img') as pbar:
            for i, data in enumerate(val_loader):
                scene_name = data['name'][0]
                if scene_name!=pre_scene:
                    frame_index = 0
                    if agent is not None:
                        agent.vis(save_graph=True)
                        # while 1: pass
                    agent = CenterLineDetAgent(args,CenterLineDet,scene_name)
                    with open(f'{args.dataroot}/label_world_crop/{scene_name}.json','r') as jf:
                        info = json.load(jf)
                    x_min, x_max, y_min, y_max = info['x_min'], info['x_max'], info['y_min'], info['y_max']
                    video_vis_map = Image.fromarray(np.zeros((y_max-y_min+1,x_max-x_min+1))).convert('RGB')
                    sat_draw = ImageDraw.Draw(video_vis_map)
                    past_vertices = []
                # get BEV segmentation map generated in advance
                pred_fused_segmentation = np.array(Image.open(os.path.join(args.savedir,f'fused_segmentation_all/pred_mask/{scene_name}_{frame_index}.png')))
                Image.open(os.path.join(args.savedir,f'fused_segmentation_all/gt_mask/{scene_name}_{frame_index}.png')).save(os.path.join(args.save_dir_single,'gt_mask',f'{scene_name}_{frame_index}.png'))
                # load info of the current patch
                _,patch_box,patch_angle,rec,idx = data['info'][0]
                segment_mask = data['segment_mask'].cuda()
                initial_candidate_mask = (np.array(Image.open(f'./{args.savedir}/fused_segmentation_all/pred_initial_candidate/{scene_name}_{frame_index}.png'))/255)>args.binary_thr
                Image.fromarray((segment_mask[0].cpu().detach().numpy()*255).astype(np.uint8)).save(f'./{args.save_dir_single}/gt_mask/{scene_name}_{frame_index}.png')
                Image.fromarray((initial_candidate_mask*255).astype(np.uint8)).save(f'./{args.savedir}/vis/single/pred_init/{scene_name}_{frame_index}.png')
                frame_index += 1
                if len(pred_fused_segmentation.shape)==3:
                    pred_fused_segmentation = pred_fused_segmentation[:,:,0]
                feature_tensor_raw = torch.Tensor(pred_fused_segmentation/255.0).unsqueeze(0).unsqueeze(0).cuda()
                # update frame info of the agent
                agent.update_frame(feature_tensor_raw,initial_candidate_mask,segment_mask,canvas_size,\
                            list(patch_box),patch_angle,frame_index)
                # agent step
                agent.step()
                # vis
                scene_vis_info = agent.scene_vis_info


                if i==len(val_loader)-1:
                    agent.vis(save_graph=True)
                pre_scene = scene_name
                pbar.update()
    
if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False 
    args = get_args()
    setup_seed(20)
    inference(args)
