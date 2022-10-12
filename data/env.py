import numpy as np
import torch
import random
import json
import cv2
import random
from shapely import affinity
import shapely.ops as so
from shapely.geometry import LineString, Polygon, box, Point
from typing import Tuple
from PIL import Image, ImageDraw
import pickle
class FrozenClass():
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True

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
            self.tracked = None
            self.init_IP = self.raw_init_IP
            self.end_IP = self.raw_end_IP

    def reverse(self):
        self.init_IP, self.end_IP = self.end_IP, self.init_IP
        self.vertices = self.vertices[::-1]
        self.orientation = self.orientation[::-1]
        self.line = LineString([tuple(x) for x in self.vertices[::-1]])

class IntersectionPoint(FrozenClass):
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
        self.neighbors = []
        self._freeze()

class Patch_LS_Intersection(FrozenClass):
    '''
    This class records the intersection of ROI patch and a line segment. 
    :param LS_inter (LineString): intersection line in shapely format (after transformation)
    :param LS (LineSegment): the corresponding LineSegment instance (before transformation)
    '''
    def __init__(self,LS_inter,LS,transformation):
        self.LS_inter = LS_inter
        self.LS = LS
        self.transformation = transformation
        self._freeze()
    
    def ego2world(self,shapely_object):
        rotation,translation,scale = self.transformation
        shapely_object = affinity.scale(shapely_object, xfact=1/scale['width'], yfact=1/scale['height'], origin=(0, 0))
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, -translation['x'], -translation['y']])
        shapely_object = affinity.rotate(shapely_object, rotation['angle'], origin=(rotation['x'],rotation['y']), use_radians=False)
        return shapely_object


class Environment(FrozenClass):
    def __init__(self,args,canvas_size,patch_size):
        self.args = args
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.agents = []
        self.MAPS_NAMES = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.centerline_maps = {}
        self.historical_map = {}
        self.train_samples = {}

        self.patch_LS_inters = []
        self.agent_index = 0
        self.previous_scene_token = ''
        self.scene_counter = 0
        self.patch_list = []
        self.previous_location = ''

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

    def update_historical_map(self,location,p1,p2):
        map = self.historical_map[location]
        map.append(LineString([tuple(p1),tuple(p2)]))

    def load_map(self):
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
        coords = np.asarray(list(lines.coords), np.int32)
        coords = coords.reshape((-1, 2))
        cv2.polylines(mask, [coords], False, color=1, thickness=2)
        return mask

    def mask_for_points(self,points, mask, rad=3):
        coords = np.asarray(list(points.coords), np.int32)
        coords = coords.reshape((-1, 2))
        cv2.circle(mask, coords[0], rad, 1, -1)
        return mask
    

    def get_centerline_mask(self,location,patch_box,patch_angle,rec,idx,save_scene_graph,save_scene_name):
        def save_scene_graph_and_vis():
            patch_x, patch_y, patch_h, patch_w = patch_box
            canvas_h = self.canvas_size[0]
            canvas_w = self.canvas_size[1]
            scale_height = canvas_h / patch_h
            scale_width = canvas_w / patch_w
            centerline_map = self.centerline_maps[self.previous_location]
            LSs = centerline_map['LSs']
            scene_patch = so.cascaded_union(self.patch_list)
            graph = Graph()
            gt_output_edges = []
            x_min, x_max, y_min, y_max = 50000,0,50000,0
            boundary_list = list(scene_patch.boundary.coords)
            x_min = int(np.floor(min([x[0] for x in boundary_list]))*scale_width)
            x_max = int(np.ceil(max([x[0] for x in boundary_list]))*scale_width)
            y_min = int(np.floor(min([x[1] for x in boundary_list]))*scale_height)
            y_max = int(np.ceil(max([x[1] for x in boundary_list]))*scale_height)

            for i,s in LSs.items():
                new_line = s.line.intersection(scene_patch)
                if not new_line.is_empty:
                    if new_line.geom_type == 'MultiLineString':
                        for new_single_line in new_line:
                            coords = np.asarray(list(new_single_line.coords),)
                            coords = [[int(x[0]*scale_height),int(x[1]*scale_width)] for x in coords]
                            gt_output_edges.append(coords)
                    else:
                        coords = np.asarray(list(new_line.coords))
                        coords = [[int(x[0]*scale_height),int(x[1]*scale_width)] for x in coords]
                        gt_output_edges.append(coords)
            vis = Image.fromarray(np.zeros((y_max-y_min+1,x_max-x_min+1))).convert('RGB')
            draw = ImageDraw.Draw(vis)
            for e in gt_output_edges:
                ii = 5
                v2 = e[0]
                while ii<len(e):
                    v1 = v2
                    v2 = e[ii]
                    ii += 5
                    graph.add([[v1[0]-x_min,v1[1]-y_min],[v2[0]-x_min,v2[1]-y_min]])
                    draw.line([v1[0]-x_min,v1[1]-y_min,v2[0]-x_min,v2[1]-y_min],fill='white',width=1)
                if ii!=len(e)-1:
                    graph.add([[e[-1][0]-x_min,e[-1][1]-y_min],[v2[0]-x_min,v2[1]-y_min]])
                    draw.line([e[-1][0]-x_min,e[-1][1]-y_min,v2[0]-x_min,v2[1]-y_min],fill='white',width=1)
            vis.save(f'{self.args.savedir}/label_gt_mask/{save_scene_name}.png')
            with open(f'{self.args.savedir}/label_world_crop/{save_scene_name}.json','w') as jf:
                json.dump({'x_min':int(x_min),'x_max':int(x_max),'y_min':int(y_min),'y_max':int(y_max)},jf)
            
            output_graph = {}
            for k, v in graph.vertices.items():
                output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

            pickle.dump(output_graph,open(f'{self.args.savedir}/label_gt_graph/{save_scene_name}.p','wb'),protocol=2)
            self.patch_list = []

        # Initialization
        self.patch_LS_inters = []
        self.agent_index = 0
        for agent in self.agents:
            agent.cease = False
        # Load corresponding map
        centerline_map = self.centerline_maps[location]
        LSs, IPs = centerline_map['LSs'], centerline_map['IPs']

        # Save gt scene graph
        if save_scene_graph==1:
            save_scene_graph_and_vis()
        self.previous_location = location
        # Patch processing
        patch_x, patch_y, patch_h, patch_w = patch_box
        
        canvas_h = self.canvas_size[0]
        canvas_w = self.canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        patch = self.get_patch_coord(patch_box, patch_angle)
        self.patch_list.append(patch)
        # Generate gt mask of centerline segments
        segment_mask = np.zeros(self.canvas_size, np.uint8)
        for i,s in LSs.items():
            new_line = s.line.intersection(patch)
            if not new_line.is_empty:
                new_line = self.world2ego(new_line,{'angle':patch_angle,'x':patch_x,'y':patch_y},\
                    {'x':trans_x,'y':trans_y},{'width':scale_width,'height':scale_height})
                if new_line.geom_type == 'MultiLineString':
                    for new_single_line in new_line:
                        self.patch_LS_inters.append(Patch_LS_Intersection(new_single_line,s,[{'angle':patch_angle,'x':patch_x,'y':patch_y},\
                        {'x':trans_x,'y':trans_y},{'width':scale_width,'height':scale_height}]))
                        segment_mask = self.mask_for_lines(new_single_line, segment_mask)
                else:
                    self.patch_LS_inters.append(Patch_LS_Intersection(new_line,s,[{'angle':patch_angle,'x':patch_x,'y':patch_y},\
                    {'x':trans_x,'y':trans_y},{'width':scale_width,'height':scale_height}]))
                    segment_mask = self.mask_for_lines(new_line, segment_mask)

        # Generate gt mask of initial candidates
        initial_candidate_mask = np.zeros(self.canvas_size,np.uint8)
        for patch_LS_inter in self.patch_LS_inters:
            initial_candidates = self.find_initial_candidates(patch_LS_inter)
            for point in initial_candidates:
                initial_candidate_mask = self.mask_for_points(point['point'], initial_candidate_mask, rad=2)
        
        
        # save
        np.savez(f'./{self.args.savedir}/label_trainval/{rec["scene_name"]}_{idx:09}.npz',rec=rec,\
                    segment_mask=segment_mask.astype(np.uint8),initial_candidate_mask=initial_candidate_mask.astype(np.uint8))
            
        self.train_samples[location] = []
        if save_scene_graph==2:
            save_scene_graph_and_vis()

    def ego2world(self,shapely_object,rotation,translation,scale):
        shapely_object = affinity.scale(shapely_object, xfact=1/scale['width'], yfact=1/scale['height'], origin=(0, 0))
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, -translation['x'], -translation['y']])
        shapely_object = affinity.rotate(shapely_object, rotation['angle'], origin=(rotation['x'],rotation['y']), use_radians=False)
        return shapely_object
    
    def world2ego(self,shapely_object,rotation,translation,scale):
        shapely_object = affinity.rotate(shapely_object, -rotation['angle'], origin=(rotation['x'],rotation['y']), use_radians=False)
        shapely_object = affinity.affine_transform(shapely_object,
                [1.0, 0.0, 0.0, 1.0, translation['x'], translation['y']])
        shapely_object = affinity.scale(shapely_object, xfact=scale['width'], yfact=scale['height'], origin=(0, 0))
        return shapely_object

    def find_initial_candidates(self,patch_LS_inter):
        endpoint1_POINT, endpoint2_POINT = patch_LS_inter.LS_inter.boundary
        endpoint1, endpoint2 = list(endpoint1_POINT.coords)[0], list(endpoint2_POINT.coords)[0]
        output = []
        if not (endpoint1[0]>0.1 and endpoint1[1]>0.1 and endpoint1[1]<self.canvas_size[0]-0.1):
            output.append({'id':1,'point':endpoint1_POINT})
        if not (endpoint2[0]>0.1 and endpoint2[1]>0.1 and endpoint2[1]<self.canvas_size[0]-0.1):
            output.append({'id':2,'point':endpoint2_POINT})
        return output