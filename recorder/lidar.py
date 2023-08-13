#!/usr/bin/python3

import re
import sys
import time
import open3d as o3d
import cv2
import math
import carla
import numpy as np

from recorder.sensor import Sensor
import tools.OpenPCDet.tools.carla_script as pcdet_tool


class Lidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save as a Nx4 numpy array. Each row is a point (x, y, z, intensity)
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.float32)
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 4), 4))

        # Convert point cloud to right-hand coordinate system
        # lidar_data[:, 1] *= -1

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        # np.save("{}/{:0>10d}".format(save_dir, sensor_data.frame), lidar_data)
        with open("{}/{:0>10d}.bin".format(save_dir,sensor_data.frame), 'wb') as file:
            file.write(lidar_data)
        return True


class SemanticLidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.load_parameters()

    def load_parameters(self):      
        # TODO: write parameters in config file(.yaml)
        self.Label_dict ={14:"Car",15:"Truck",16:"Bus"}
        self.Active = True
        self.Largest_label_range = 75       # object labeling range, max 100
            
        if self.Active:
            self._Hs = 0.8                  # scene entropy 

            self._rho_b = 25                # temp object rho 
            self._rho_s = 5                # temp object rho 
            self._f_tra = 1.2                 # tracking task 

            self._k_sig = 4                 # parameter value sigimoid 
            self._f_sig = 0.8               # result value sigimoid

            self.last_entropy = 3
 
            # self.load_object_detection_model()

            self.last_trans_dict = {}
            


    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save data as a Nx6 numpy array.
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('CosAngle', np.float32),
                                       ('ObjIdx', np.uint32),
                                       ('ObjTag', np.uint32)
                                   ]))

        # Convert point cloud to right-hand coordinate system
        # lidar_data['y'] *= -1

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy

        if self.Active:
            print("use active strategy")
            # result, labels, score = self.active_manager(lidar_data)
            result, labels, score = self.one_loop_cal_all_active(lidar_data)
            if result:
                self.save_data(save_dir,sensor_data,lidar_data,labels)

        else:
            labels,_ = self.get_label(lidar_data)
            self.save_data(save_dir,sensor_data,lidar_data,labels)
        return True
    

    def save_data(self,save_dir,sensor_data,lidar_data,labels):
        with open("{}/{:0>10d}.bin".format(save_dir,sensor_data.frame), 'wb') as file:
                file.write(lidar_data)
        with open("{}/{:0>10d}.txt".format(save_dir,sensor_data.frame),'a+') as f:
            for line in labels:
                print(line,file=f)

    def get_label(self,lidar_data):
        labels = []
        objects_dict,ground = self.get_label_centerpoint(lidar_data)
        if not self.Active:
            bbox_dict,trans_dict,tags_dict,sensor_trans = self.get_near_boudning_box_by_world()
        else:
            bbox_dict,trans_dict,tags_dict,sensor_trans = self.get_area_point_rho_objects(objects_dict)
        for key in bbox_dict:
            if key in objects_dict.keys():
                temp_bbox = bbox_dict[key]
                temp_points = objects_dict[key]
                temp_points = np.array([list(elem) for elem in temp_points])

                max_p = np.max(temp_points, axis=0)
                min_p = np.min(temp_points, axis=0)
                temp_bbox = bbox_dict[key]
                temp_trans = trans_dict[key]
                temp_tag = tags_dict[key]

                np.mean(max_p[0] + min_p[0])
                cx = (max_p[0] + min_p[0])/2
                cy = (max_p[1] + min_p[1])/2
                cz = (temp_trans.location.z - sensor_trans.location.z +temp_bbox.location.z)

                sx = 2*temp_bbox.extent.x
                sy = 2*temp_bbox.extent.y
                sz = 2*temp_bbox.extent.z
                yaw = (temp_trans.rotation.yaw - sensor_trans.rotation.yaw + temp_bbox.rotation.yaw)

                label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[temp_tag[0]])
    
                labels.append(label_str)
        
        return labels,ground
    
    def get_label_centerpoint(self,semantic_points):
        objects_dict = {}
        ground = []
        for point in semantic_points:
            if point[5] in self.Label_dict.keys():
                if not point[4] in objects_dict:
                    objects_dict[point[4]] = []
                objects_dict[point[4]].append(point)
            elif point[5] == 1:
                ground.append(point)


        return objects_dict,ground

    def set_world(self, world):
        self.world = world
        

    def get_near_boudning_box_by_world(self):
        bbox_dict = {}
        trans_dict={}
        tags_dict = {}

        actors_list = self.world.get_actors()
        for actor in actors_list:
            if re.match("^vehicle",str(actor.type_id)):
                dist = actor.get_transform().location.distance(self.carla_actor.get_transform().location)
                if dist < self.Largest_label_range:
                    bbox_dict[actor.id] = actor.bounding_box
                    trans_dict[actor.id] = actor.get_transform()
                    tags_dict[actor.id] = actor.semantic_tags
        
        return bbox_dict,trans_dict,tags_dict,self.carla_actor.get_transform()
    
    # ============== Active Startegy ===================

    def active_manager(self,lidar_data):

        # L3-1. Scene Entropy
        tick = time.time()
        lidar_data = np.array([list(elem) for elem in lidar_data])
        entropy_now = self.get_scene_entropy(lidar_data[:,:3])
        print("#########################################entropy time: ", time.time() - tick)


        if self.last_entropy == 3:
            self.last_entropy = entropy_now
        elif self.cal_entropy_if_keep(self.last_entropy, entropy_now) or entropy_now < -20000:
            self.last_entropy = entropy_now
        else:
            print("##########################################entropy not change", entropy_now)
            return False, None, None
        
        # L3-2. Area point rho
        labels, ground_points = self.get_label()

        frequency_score = 0

        # # L4-1.1 max detecting distance
        # # ground_points = self.cal_segmentated_ground(lidar_data)
    
        # max_dist = self.get_max_detecting_distance(ground_points, labels[:,:3])

        # # L4-1.2 valid detecting distance
        # predict_result = self.cal_detect_result(lidar_data)
        # det_dist = self.get_detecting_distance(labels, predict_result)

        # dist_score = self.cal_sigmoid(max_dist, det_dist)
        
        # # L4-2. detecting precision
        # precision_score = self.get_detecting_precision(predict_result, labels)
        
        # frequency_score = dist_score * precision_score

        return True, labels, frequency_score



    # ------------- 1. Scene Entropy -------------------

    def get_scene_entropy(self, points):
        print("###NOW GET SCENE ENTROPY###")
        voxel_size = 2
        voxel_max_range = np.max(points, axis=0)
        voxel_min_range = np.min(points, axis=0)

        voxel_count = [int((voxel_max_range[0] - voxel_min_range[0])/voxel_size),
                      int((voxel_max_range[1]-voxel_min_range[1])/voxel_size),
                      int((voxel_max_range[2]-voxel_min_range[2])/voxel_size)]
        
        print("###VOXEL COUNT INIT###")

        
        voxel_scene = np.zeros(voxel_count)
        for point in points:
            voxel_scene[int((point[0] - voxel_min_range[0])/voxel_size)][int((point[1] - voxel_min_range[1])/voxel_size)][int((point[2] - voxel_min_range[2])/voxel_size)] += 1

        print("###VOXEL COUNT FINISH###")

        dt = len(points) / ((voxel_max_range[0] - voxel_min_range[0]) * (voxel_max_range[1] - voxel_min_range[1]) * (voxel_max_range[2] - voxel_min_range[2]))
        entropy = 0
        for i in range(voxel_count[0]):
            for j in range(voxel_count[1]):
                for k in range(voxel_count[2]):
                    di = voxel_scene[i][j][k]
                    entropy -= (di / dt) *  math.log10(di / dt)

        print("###ENTROPY FINISH###")

        return entropy
    
    def cal_entropy_if_keep(self, entropy_last, entropy_now):
        return abs(entropy_now - entropy_last) / -entropy_last >= self._Hs

    # ------------- 2. Area point rho ------------------

    def get_area_point_rho_objects(self, obj_dict):
        bbox_dict = {}
        trans_dict={}
        tags_dict = {}

        actors_list = self.world.get_actors()
        for actor in actors_list:
            if actor.id in obj_dict.keys() and re.match("^vehicle",str(actor.type_id)):
                dist = actor.get_transform().location.distance(self.carla_actor.get_transform().location)
                
                if dist < self.Largest_label_range:
                    bbox = actor.bounding_box
                    count = len(obj_dict[actor.id])
                    volume = 8 * bbox.extent.x * bbox.extent.y * bbox.extent.z
                    rho = count / volume

                    if rho > self._rho_b:
                        # upper big matric
                        bbox_dict[actor.id] = actor.bounding_box
                        trans_dict[actor.id] = actor.get_transform()
                        tags_dict[actor.id] = actor.semantic_tags
                    elif rho > self._rho_s:
                        # upper small matric, calculate speed
                        velocity = actor.get_velocity()
                        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                        if speed > self._f_tra:
                            bbox_dict[actor.id] = actor.bounding_box
                            trans_dict[actor.id] = actor.get_transform()
                            tags_dict[actor.id] = actor.semantic_tags
                        pass

        return bbox_dict, trans_dict, tags_dict, self.carla_actor.get_transform()

    # ------------- 3. Detecting Distance --------------

    def load_ground_segmentation_model(self):
        try:
            seg_module_path= "./tools/patchwork-plusplus/build/python_wrapper"
            sys.path.insert(0, seg_module_path)
            import pypatchworkpp 

        except ImportError:
            print("Cannot find Segmentation Module! Maybe you should build it first.")
            print("See more details in utils/patchwork-plusplus/README.md")
            exit(1)

        params = pypatchworkpp.Parameters()
        self.seg_module = pypatchworkpp.patchworkpp(params)
        

    def cal_segmentated_ground(self,lidar_data):
        self.seg_module.estimateGround(lidar_data[:,:4])
        return self.seg_module.getGround()
    
    def get_max_detecting_distance(self, ground_points, vehicle_center):
        # get the max distance from vehicle center to the ground points
        # vehicle_center is the list of labeled vehicle center points
        rect = cv2.minAreaRect(ground_points[:,:2])
        rect_points = cv2.boxPoints(rect)
        rect_points = np.array(rect_points)
        A = rect_points[0]
        B = rect_points[1]
        C = rect_points[2]
        D = rect_points[3]
        rect_center = rect[0]
        dist = 0
        
        for p in vehicle_center:
            # vector direction, if (BA*Bp)(DC*Dp)>=0, cross first, then judge the direction by the sign of cross product
            if np.cross(B-A,p-B) * np.cross(D-C,p-D) >= 0 and np.cross(C-B,p-C) * np.cross(A-D,p-A) >= 0:
                tmp_dist = math.sqrt((p[0]-rect_center[0])**2 + (p[1]-rect_center[1])**2)
                if tmp_dist > dist:
                    dist = tmp_dist
        
        return dist
    
    def load_object_detection_model(self):
        self.model_tool = pcdet_tool.PCDmodel()
        self.model_tool.load_model("pointpillar")

        return
    
    def cal_detect_result(self,path):
        data_dict = self.model_tool.load_per_data(path)
        predict_result = self.model_tool.model.forward(data_dict)

        return predict_result
    
    def get_detecting_distance(self,labels,predict_result):
        pred_boxes = predict_result[0]['pred_boxes'].cpu().numpy() # N*7
        pred_scores = predict_result[0]['pred_scores'].cpu().numpy() # N
        pred_labels = predict_result[0]['pred_labels'].cpu().numpy() # N

        dist = 0
        for i in range(len(pred_boxes)):
            if pred_scores[i] > 0.6:
                temp= math.sqrt(pred_boxes[i][0]**2 + pred_boxes[i][1]**2 + pred_boxes[i][2]**2)
                dist = max(dist, temp)
            return dist

    def cal_sigmoid(self, max_dist, det_dist):
        x = det_dist/ max_dist
        y = (1 - np.power(np.e, -self._k_sig * x) )/(1 + np.power(np.e, -self._k_sig * x))

        return 1 - y 

    # ------------- 4. Detecting Precision -------------

    def get_detecting_precision(self, predict_result, GT_result):
        pred_boxes = predict_result[0]['pred_boxes'].cpu().numpy() # N*7
        pred_scores = predict_result[0]['pred_scores'].cpu().numpy() # N
        pred_labels = predict_result[0]['pred_labels'].cpu().numpy() # N

        # GT_result format:
        # list: [x, y, z, w, l, h, yaw, label]
        # label: 1:Car, 2:Pedestrian, 3:Cyclist

        # calculate the precision of each object



        return 100


    # ============== Active Startegy ===================

    def cal_BEV_Heatmap(self,points):
        # calculate the max depth in BEV, and the max depth in the whole scene

        return
    
    def cal_BEV_Heatmap_entropy(self):

        return
    


    def one_loop_cal_all_active(self,lidar_data):
        # 1. Initialize lidar_data
        tick = time.time()
        semantic_points = np.array([list(elem) for elem in lidar_data])

        # 2. Calculate the scene entropy
        voxel_size = 2
        voxel_max_range = np.max(semantic_points[:,:3], axis=0)
        voxel_min_range = np.min(semantic_points[:,:3], axis=0)

        
        # 3. Get label center point
        objects_dict = {}
        entropy = 0

        voxel_count = np.ceil((voxel_max_range - voxel_min_range) / voxel_size).astype(int)[:3]
        voxel_scene = np.zeros(voxel_count)
        
        indices = np.floor((semantic_points[:,:3] - voxel_min_range) / voxel_size).astype(int)
        for i in indices:
            voxel_scene[i[0],i[1],i[2]] += 1
        # np.add.at(voxel_scene, (indices[:,0], indices[:,1], indices[:,2]), 1)
        


        dt = len(semantic_points) / ((voxel_max_range[0] - voxel_min_range[0]) * (voxel_max_range[1] - voxel_min_range[1]) * (voxel_max_range[2] - voxel_min_range[2]))

        nonzero_indices = np.nonzero(voxel_scene)
        di = voxel_scene[nonzero_indices]
        entropy = -np.sum((di / dt) * np.log10(di / dt))

        valid_labels = np.isin(semantic_points[:, 5], list(self.Label_dict.keys()))
        valid_points = semantic_points[valid_labels]
        unique_labels = np.unique(valid_points[:, 4])

        for label in unique_labels:
            objects_dict[int(label)] = valid_points[valid_points[:, 4] == label]
        
        ground_indices = np.where(semantic_points[:, 5] == 1)
        ground = semantic_points[ground_indices]
        
        if self.last_entropy == 3:
            self.last_entropy = entropy
        elif self.cal_entropy_if_keep(self.last_entropy, entropy) or entropy < -20000:
            self.last_entropy = entropy
        else:
            return False, None, None

            
        # 4. Calculate the area point rho
        
        labels = []
        now_trans_dict = {}

        actors_list = self.world.get_actors()
        for actor in actors_list:
            if actor.id in objects_dict.keys() and re.match("^vehicle",str(actor.type_id)):
                dist = actor.get_transform().location.distance(self.carla_actor.get_transform().location)

                if dist < self.Largest_label_range:
                    bbox = actor.bounding_box
                    points_collection = objects_dict[actor.id]
                    count = len(points_collection)
                    volume = 8 * bbox.extent.x * bbox.extent.y * bbox.extent.z
                    rho = count / volume

                    if rho > self._rho_b:

                        points_collection = np.array([list(elem) for elem in points_collection])
                        max_p = np.max(points_collection, axis=0)
                        min_p = np.min(points_collection, axis=0)
                        cx = (max_p[0] + min_p[0])/2
                        cy = (max_p[1] + min_p[1])/2
                        cz = (actor.get_transform().location.z - self.carla_actor.get_transform().location.z + bbox.location.z)

                        sx = 2*bbox.extent.x
                        sy = 2*bbox.extent.y
                        sz = 2*bbox.extent.z
                        yaw = (actor.get_transform().rotation.yaw - self.carla_actor.get_transform().rotation.yaw + bbox.rotation.yaw)

                        label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
                        labels.append(label_str)
                        now_trans_dict[actor.id] = [cx,cy]
                        
                    elif rho > self._rho_s:
                        points_collection = np.array([list(elem) for elem in points_collection])
                        max_p = np.max(points_collection, axis=0)
                        min_p = np.min(points_collection, axis=0)
                        cx = (max_p[0] + min_p[0])/2
                        cy = (max_p[1] + min_p[1])/2

                        now_trans_dict[actor.id] = [cx,cy]
                        if actor.id in self.last_trans_dict.keys():
                      
                            delta_trans = [self.last_trans_dict[actor.id][0]-cx, self.last_trans_dict[actor.id][1]-cy]
                            delta_dist = np.sqrt(delta_trans[0]**2 + delta_trans[1]**2)
                        
                            if delta_dist > self._f_tra:
                                cz = (actor.get_transform().location.z - self.carla_actor.get_transform().location.z + bbox.location.z)

                                sx = 2*bbox.extent.x
                                sy = 2*bbox.extent.y
                                sz = 2*bbox.extent.z
                                yaw = (actor.get_transform().rotation.yaw - self.carla_actor.get_transform().rotation.yaw + bbox.rotation.yaw)

                                label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
                                labels.append(label_str)
                                
                            else:
                                print("[delta_dist]", delta_dist)
        
        self.last_trans_dict = now_trans_dict
        print("###[LABEL FINISH]###", time.time() - tick,"###[entropy]###", entropy)
        
        return True, labels, 100
    

    def one_loop_update_frequency(self,ground,path,labels):
        # 1. Calculate max detecting distance

        # 2. Calculate valid detecting distance

        # 3. Calculate two distance score

        # 4. Calculate detecting precision

        return

