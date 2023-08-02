#!/usr/bin/python3

import re
import asyncio
import open3d as o3d
import cv2
import carla
import numpy as np
import label_tools.kitti_lidar.lidar_label_view as label_tool
import label_tools.lidar_tool.util as util
from recorder.sensor import Sensor


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
        self.dis_dict = {}

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
        # dataset, now_dis, score = label_tool.save_label(lidar_data, self.dis_dict)
        # self.dis_dict = now_dis

        labels = self.get_label(lidar_data)

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
        objects_dict = self.get_label_centerpoint(lidar_data)
        bbox_dict,trans_dict,tags_dict,sensor_trans = self.get_near_boudning_box_by_world()
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

                cx = (max_p[0] + min_p[0])/2
                cy = (max_p[1] + min_p[1])/2
                cz = (temp_trans.location.z - sensor_trans.location.z +temp_bbox.location.z)

                sx = 2*temp_bbox.extent.x
                sy = 2*temp_bbox.extent.y
                sz = 2*temp_bbox.extent.z
                rotation_y = (temp_trans.rotation.yaw - sensor_trans.rotation.yaw + temp_bbox.rotation.yaw)

                label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, rotation_y, "Car-" + str(key) + str(temp_tag))
    
                labels.append(label_str)
        
        return labels
    
    def get_label_centerpoint(self,semantic_points):
        usable_labels = {14,15,16}
        objects_dict = {}
        for point in semantic_points:
            if point[5] in usable_labels:
                if not point[4] in objects_dict:
                    objects_dict[point[4]] = []
                objects_dict[point[4]].append(point)

        return objects_dict
    
    def set_car_list(self, record_cars, other_cars, world):
        self.record_cars = record_cars
        self.other_cars = other_cars
        self.world = world


    def set_world(self, world):
        self.world = world
        

    def get_near_boudning_box_by_world(self):
        actors_list = self.world.get_actors()

        bbox_dict = {}
        trans_dict={}
        tags_dict = {}
     
        for actor in actors_list:
            if re.match("^vehicle",str(actor.type_id)):
                dist = actor.get_transform().location.distance(self.carla_actor.get_transform().location)
                if dist < 80:
                    bbox_dict[actor.id] = actor.bounding_box
                    trans_dict[actor.id] = actor.get_transform()
                    tags_dict[actor.id] = actor.semantic_tags
        
        return bbox_dict,trans_dict,tags_dict,self.carla_actor.get_transform()
