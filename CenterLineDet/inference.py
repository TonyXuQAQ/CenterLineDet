import os
import numpy as np
import sys
import argparse
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import measure
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Polygon, box, Point
import json
import pickle

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
        new_v = self.agent.ego2world(Point(new_v))
        if self.linestring is None:
            self.linestring = LineString(list(self.first_vertex.coords)+list(new_v.coords))
        else:
            self.linestring = LineString(list(self.linestring.coords)+list(new_v.coords))
        self.last_vertex = Point(list(self.linestring.coords)[-1])
        
    
    def add_reverse_v(self,new_v):
        new_v = self.agent.ego2world(Point(new_v))
        if self.linestring is None:
            self.linestring = LineString(list(new_v.coords) + list(self.first_vertex.coords))
        else:
            self.linestring = LineString(list(new_v.coords)+list(self.linestring.coords))
        self.first_vertex = Point(list(self.linestring.coords)[0])


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
        # world
        self.world_historical_lines = []
        self.world_initial_candidates = []
        self.world_point_previous_frame = []
        self.world_instances = []
        
        self.gt_points = []

        # counter 
        self.sample_counter = 0
        self.step_counter = 0
        self.instance_counter = 0
        
    def update_frame(self,pred_segment_mask,initial_candidate_mask,segment_mask,canvas_size,patch_box,patch_angle):
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
        self.segment_mask = segment_mask
        self.step_counter = 0
        self.rotation = {'angle':patch_angle,'x':patch_x,'y':patch_y}
        self.translation = {'x':trans_x,'y':trans_y}
        self.scale = {'width':scale_width,'height':scale_height}
        self.patch = self.get_patch_coord([patch_x, patch_y, patch_h, patch_w],patch_angle)
        self.frame_historical_map = Image.fromarray(np.zeros((200,200)))
        self.sample_counter += 1
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
                Center_world = list(self.ego2world(center_point).coords)[0]
                self.world_initial_candidates.insert(0,Center_world)

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
        :param v_previous: previous vertex
        :param pred_logits: logits of predicted coordinates
        :param pred_coords: predicted vertex coordinates (translation of v_current)
        :param thr: threshold for pred_logits
        :return output: extracted valid coords
        '''
        pred_coord = pred_coord[0]
        pred_coord = pred_coord.cpu().detach().numpy().tolist()
        pred_coord = [[x[0]*self.args.ROI_half_length+v_current[0],x[1]*self.args.ROI_half_length+v_current[1]] for x in pred_coord]
        pred_logits = pred_logits[0].softmax(dim=1)
        valid_vertices = []
        for ii, coord in enumerate(pred_coord):
            if pred_logits[ii,0] >= thr and coord[0]>=0 and coord[0]<200 and coord[1]>=0 and coord[1]<200:
                valid_vertices.append(coord)
        # previous vector
        vector_previous = np.array(v_current) - np.array(v_previous)
        norm_previous = np.linalg.norm(vector_previous)
        # filter by angle
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
        return output

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
            inter_line = self.world2ego(inter_line)
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
            frame_initial_candidates.append(list(self.world2ego(Point(world_initical_candidate)).coords)[0])
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
            new_instance = CenterLineInstance(self,self.ego2world(Point(frame_initial_candidate)))
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
        world_v1 = list(self.ego2world(Point(frame_v1)).coords)[0]
        world_v2 = list(self.ego2world(Point(frame_v2)).coords)[0]
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
                v_next_world = list(self.ego2world(Point(v_next)).coords)[0]
                first_vertex = list(instance.first_vertex.coords)[0]
                if np.linalg.norm(np.array(v_next_world)-np.array(first_vertex)) <self.args.world_alignment_distance:
                    v_next = list(self.world2ego(Point(first_vertex)).coords)[0]
                    return True, v_next
                
                last_vertex = list(instance.last_vertex.coords)[0]
                if np.linalg.norm(np.array(v_next_world)-np.array(last_vertex)) <self.args.world_alignment_distance:
                    v_next = list(self.world2ego(Point(last_vertex)).coords)[0]
                    return True, v_next
        return False, v_next
        
    def step(self):
        '''
        Step function of the agent
        '''
        self.world_objects_to_local()
        draw_history_map = ImageDraw.Draw(self.frame_historical_map)
        
        while len(self.frame_instances)and self.step_counter<200:
            if len(self.frame_instances):
                instance = self.frame_instances.pop(0)
                self.removed_frame_instance.append(instance)
            else:
                break
            # iterations
            # add from last vertex of the instance
            if instance.direction_add and not instance.remove_last_vertex:
                v_current = [x for x in list(self.world2ego(instance.last_vertex).coords)[0]]
                v_previous = v_current
                instance_counter = 0
                # iterations
                while instance_counter < 40:
                    instance_counter += 1
                    self.step_counter += 1
                    # Process feature tensor      
                    feature_tensor = self.crop_tensor(v_current)
                    # predict vertices in the next step
                    action_pred = self.net(feature_tensor[:,[0,2]])
                    pred_coords = self.extract_valid_coords(v_current,v_previous,action_pred['pred_logits'],action_pred['pred_boxes'],thr=self.args.logit_thr)
                    v_previous = v_current
                    # take corresponding actions based on the number of valid predicted vertices
                    if not len(pred_coords):
                        break
                    elif len(pred_coords) == 1:
                        draw_history_map.line([v_current[0],v_current[1],pred_coords[0][0],pred_coords[0][1]],fill='white',width=1)
                        stop_action, pred_coords[0] = self.alignment(instance,pred_coords[0])
                        self.add_line(v_current,pred_coords[0])
                        v_current = pred_coords[0]
                        instance.add_v(v_current)
                        if stop_action:
                            instance.remove_first_vertex = True
                            break
                        if instance_counter>=40 or v_current[0]>198 or v_current[1]>198 or v_current[1]<5 :
                            break
                    else:
                        instance.remove_last_vertex = True
                        for idx in range(len(pred_coords)):
                            stop_action, pred_coords[0] = self.alignment(instance,pred_coords[idx])
                            self.add_line(v_current,pred_coords[idx])
                            draw_history_map.line([v_current[0],v_current[1],pred_coords[idx][0],pred_coords[idx][1]],fill='white',width=1)
                            # add new instance 
                            if not stop_action:
                                new_instance = CenterLineInstance(self,self.ego2world(Point(v_current)))
                                new_instance.add_v(pred_coords[idx])
                                new_instance.remove_first_vertex = True
                                self.world_instances.append(new_instance)
                                self.frame_instances.append(new_instance)
                        break
            # add from first vertex of the instance
            if instance.reverse_direction_add and not instance.remove_first_vertex:
                v_current = [(x) for x in list(self.world2ego(instance.first_vertex).coords)[0]]
                v_previous = v_current
                instance_counter = 0
                # iterations
                while instance_counter < 40:
                    instance_counter += 1
                    self.step_counter += 1
                    # Process feature tensor      
                    feature_tensor = self.crop_tensor(v_current)
                    # predict vertices in the next step
                    action_pred = self.net(feature_tensor[:,[0,2]])
                    pred_coords = self.extract_valid_coords(v_current,v_previous,action_pred['pred_logits'],action_pred['pred_boxes'],thr=self.args.logit_thr)
                    v_previous = v_current
                    # take corresponding actions based on the number of valid predicted vertices
                    if not len(pred_coords):
                        break
                    elif len(pred_coords) == 1:
                        stop_action, pred_coords[0] = self.alignment(instance,pred_coords[0])
                        draw_history_map.line([v_current[0],v_current[1],pred_coords[0][0],pred_coords[0][1]],fill='white',width=1)
                        self.add_line(v_current,pred_coords[0])
                        v_current = pred_coords[0]
                        instance.add_reverse_v(v_current)
                        if stop_action:
                            instance.remove_last_vertex = True
                            break
                        if instance_counter>=40 or v_current[0]>198 or v_current[1]>198 or v_current[1]<5 :
                            break
                    else:
                        instance.remove_first_vertex = True
                        for idx in range(len(pred_coords)):
                            stop_action, pred_coords[0] = self.alignment(instance,pred_coords[idx])
                            self.add_line(v_current,pred_coords[idx])
                            draw_history_map.line([v_current[0],v_current[1],pred_coords[idx][0],pred_coords[idx][1]],fill='white',width=1)
                            # add new instance 
                            if not stop_action:
                                new_instance = CenterLineInstance(self,self.ego2world(Point(v_current)))
                                new_instance.add_v(pred_coords[idx])
                                new_instance.remove_first_vertex = True
                                self.world_instances.append(new_instance)
                                self.frame_instances.append(new_instance)
                        break
        self.frame_historical_map.convert('RGB').save(os.path.join(self.args.save_dir_single,'pred_mask',f'{self.scene_name}_{self.sample_counter-1}.png'))
           
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
        
        if save_graph:
            graph = Graph()
            for e in output_edges:
                graph.add(e)

            output_graph = {}
            for k, v in graph.vertices.items():
                output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

            pickle.dump(output_graph,open(os.path.join(self.args.save_dir_multi,'pred_graph',f'{self.scene_name}.p'),'wb'),protocol=2)

        
def inference(args):

    # ============== 
    args.save_dir_single = f'./{args.savedir}/inference/single'
    create_directory(f'./{args.savedir}/inference/single/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/inference/single/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/inference/single/pred_init',delete=True)

    args.save_dir_multi = f'./{args.savedir}/inference/multi'
    create_directory(f'./{args.savedir}/inference/multi/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/inference/multi/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/inference/multi/pred_graph',delete=True)
    create_directory(f'./{args.savedir}/inference/multi/pred_csv',delete=True)

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

    val_loader = inference_centerlinedet_dataset(args, data_conf)
    
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
                # get BEV segmentation map generated in advance
                pred_fused_segmentation = np.array(Image.open(os.path.join(args.savedir,f'fused_segmentation_all/pred_mask/{scene_name}_{frame_index}.png')))
                Image.open(os.path.join(args.savedir,f'fused_segmentation_all/gt_mask/{scene_name}_{frame_index}.png')).save(os.path.join(args.save_dir_single,'gt_mask',f'{scene_name}_{frame_index}.png'))
                # load info of the current patch
                _,patch_box,patch_angle,rec,idx = data['info'][0]
                segment_mask = data['segment_mask'].cuda()
                initial_candidate_mask = (np.array(Image.open(f'./{args.savedir}/fused_segmentation_all/pred_initial_candidate/{scene_name}_{frame_index}.png'))/255)>args.binary_thr
                Image.fromarray((segment_mask[0].cpu().detach().numpy()*255).astype(np.uint8)).save(f'./{args.save_dir_single}/gt_mask/{scene_name}_{frame_index}.png')
                Image.fromarray((initial_candidate_mask*255).astype(np.uint8)).save(f'./{args.savedir}/inference/single/pred_init/{scene_name}_{frame_index}.png')
                frame_index += 1
                if len(pred_fused_segmentation.shape)==3:
                    pred_fused_segmentation = pred_fused_segmentation[:,:,0]
                feature_tensor_raw = torch.Tensor(pred_fused_segmentation/255.0).unsqueeze(0).unsqueeze(0).cuda()
                # update frame info of the agent
                agent.update_frame(feature_tensor_raw,initial_candidate_mask,segment_mask,canvas_size,\
                            list(patch_box),patch_angle)
                # agent step
                agent.step()
                if i==len(val_loader)-1:
                    agent.vis(save_graph=True)
                pre_scene = scene_name
                pbar.update()
                    
def evaluate(args,data_conf,CenterLineDet,val_loader):

    def calculate_scores(gt_points,pred_points):
        gt_tree = cKDTree(gt_points)
        if len(pred_points):
            pred_tree = cKDTree(pred_points)
        else:
            return 0,0,0
        thr = 3
        dis_gt2pred,_ = pred_tree.query(gt_points, k=1)
        dis_pred2gt,_ = gt_tree.query(pred_points, k=1)
        recall = len([x for x in dis_gt2pred if x<thr])/len(dis_gt2pred)
        acc = len([x for x in dis_pred2gt if x<thr])/len(dis_pred2gt)
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
        output.append(pixel_eval_metric(pred_mask,gt_mask))
        return output
        
    # ============== 
    args.save_dir_single = f'./{args.savedir}/val/single'
    create_directory(f'./{args.savedir}/val/single/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/val/single/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/val/single/pred_init',delete=True)

    args.save_dir_multi = f'./{args.savedir}/val/multi'
    create_directory(f'./{args.savedir}/val/multi/gt_mask',delete=True)
    create_directory(f'./{args.savedir}/val/multi/pred_mask',delete=True)
    create_directory(f'./{args.savedir}/val/multi/pred_csv',delete=True)
    
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
    patch_size = (patch_h, patch_w)
    canvas_size = (canvas_h, canvas_w)

    # model.eval()
    CenterLineDet.eval()
    
    pre_scene = None
    agent = None
    frame_index = 0
    with torch.no_grad():
        with tqdm(total=len(val_loader), unit='img') as pbar:
            for i, data in enumerate(val_loader):
                scene_name = data['name'][0]
                if scene_name!=pre_scene :
                    pre_scene = scene_name
                    frame_index = 0
                    if agent is not None:
                        agent.vis()
                    agent = CenterLineDetAgent(args,CenterLineDet,scene_name)
                # get BEV segmentation map generated in advance
                pred_fused_segmentation = np.array(Image.open(os.path.join(args.savedir,f'fused_segmentation_all/pred_mask/{scene_name}_{frame_index}.png')))
                # load info of the current patch
                _,patch_box,patch_angle,rec,idx = data['info'][0]
                segment_mask = data['segment_mask'].cuda()#.cpu().detach().numpy()[0]
                initial_candidate_mask = (np.array(Image.open(f'./{args.savedir}/fused_segmentation_all/pred_initial_candidate/{scene_name}_{frame_index}.png'))/255)>args.binary_thr
                Image.fromarray((initial_candidate_mask*255).astype(np.uint8)).save(f'./{args.savedir}/val/single/pred_init/{scene_name}_{frame_index}.png')
                frame_index += 1
                if len(pred_fused_segmentation.shape)==3:
                    pred_fused_segmentation = pred_fused_segmentation[:,:,0]
                feature_tensor_raw = torch.Tensor(pred_fused_segmentation/255.0).unsqueeze(0).unsqueeze(0).cuda()
                # update frame info of the agent
                agent.update_frame(feature_tensor_raw,initial_candidate_mask,segment_mask,canvas_size,\
                            list(patch_box),patch_angle)
                # agent step
                agent.step()
                # if i==len(val_loader)-1:
                #     agent.vis()
                pre_scene = scene_name
                pbar.update()
                
    score_list = []
    for file_name in os.listdir(os.path.join(args.save_dir_multi,'pred_mask')):
        pred_mask = np.array(Image.open(os.path.join(args.save_dir_multi,'pred_mask',file_name)))
        if len(pred_mask.shape)==3:
            pred_mask = pred_mask[:,:,0]
        gt_mask = np.array(Image.open(os.path.join('../dataset/label_gt_mask',file_name)))
        if len(gt_mask.shape)==3:
            gt_mask = gt_mask[:,:,0]
        score_list.extend(get_score(pred_mask,gt_mask))
    return sum([x[0] for x in score_list])/(len(score_list)+1e-7),sum([x[1] for x in score_list])/(len(score_list)+1e-7),sum([x[2] for x in score_list])/(len(score_list)+1e-7)


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False 
    args = get_args()
    setup_seed(20)
    inference(args)
