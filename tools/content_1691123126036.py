from typing import List
from math import pi
from math import atan
import os
import csv
import numpy as np

input_folder = "carladata"
output_folder = "kittidata"

class KittiDescriptor:
    # This class is responsible for storing a single datapoint for the kitti 3d object detection task
    def __init__(self, type=None, bbox=None, dimensions=None, location=None, rotation_y=None, extent=None):
        self.type = type
        self.truncated = 0
        self.occluded = 0
        self.alpha = -10
        self.bbox = bbox
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.extent = extent
        self._valid_classes = ['Car', 'Van', 'Truck',
                               'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                               'Misc', 'DontCare']

    def set_type(self, obj_type: str):
        assert obj_type in self._valid_classes, "Object must be of types {}".format(
            self._valid_classes)
        self.type = obj_type

    def set_truncated(self, truncated: float):
        assert 0 <= truncated <= 1, """Truncated must be Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries """
        self.truncated = truncated

    def set_occlusion(self, occlusion: int):
        assert occlusion in range(0, 4), """Occlusion must be Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown"""
        self._occluded = occlusion

    def set_alpha(self, alpha: float):
        # assert -pi <= alpha <= pi, "Alpha must be in range [-pi..pi]"
        self.alpha = alpha

    def set_bbox(self, bbox: List[int]):
        assert len(bbox) == 4, """ Bbox must be 2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates (two points)"""
        self.bbox = bbox

    def set_3d_object_dimensions(self, x, y, z):
        # Bbox extent consists of x,y and z.
        # The bbox extent is by Carla set as
        # x: length of vehicle (driving direction)
        # y: to the right of the vehicle
        # z: up (direction of car roof)
        # However, Kitti expects height, width and length (z, y, x):
        # Since Carla gives us bbox extent, which is a half-box, multiply all by two
        height = z
        width = y
        length = x
        self.extent = (height, width, length)
        self.dimensions = "{} {} {}".format(2*height, 2*width, 2*length)

    def set_3d_object_location(self, x, y, z):
        """ TODO: Change this to 
            Converts the 3D object location from CARLA coordinates and saves them as KITTI coordinates in the object
            In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
            z
            ▲   ▲ x
            |  /
            | /
            |/____> y
            This is a left-handed coordinate system, with x being forward, y to the right and z up 
            See also https://github.com/carla-simulator/carla/issues/498
            However, the camera coordinate system for KITTI is defined as
                ▲ z
               /
              /
             /____> x
            |
            |
            |
            ▼
            y 
            This is a right-handed coordinate system with z being forward, x to the right and y down
            Therefore, we have to make the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI:-X  -Y   Z
        """

        # Convert from Carla coordinate system to KITTI
        # This works for AVOD (image)
        #x *= -1
        #y *= -1
        #self.location = " ".join(map(str, [y, -z, x]))
        self.location = " ".join(map(str, [-x, -y, z]))
        # This works for SECOND (lidar)
        #self.location = " ".join(map(str, [z, x, y]))
        #self.location = " ".join(map(str, [z, x, -y]))

    def set_rotation_y(self, rotation_y: float):
        # assert - \
        #     pi <= rotation_y <= pi, "Rotation y must be in range [-pi..pi] - found {}".format(
        #         rotation_y)
        self.rotation_y = rotation_y

    def __str__(self):
        """ Returns the kitti formatted string of the datapoint if it is valid (all critical variables filled out), else it returns an error."""
        if self.bbox is None:
            bbox_format = " "
        else:
            bbox_format = " ".join([str(x) for x in self.bbox])

        # return "{} {} {} {} {} {} {} {}".format(self.type, self.truncated, self.occluded, 
        #                                         self.alpha, bbox_format, self.dimensions, 
        #                                         self.location, self.rotation_y)
        return "{} {} {} {}".format(self.location, self.dimensions, 
                                       self.rotation_y, self.type)


input_files = os.listdir(input_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in input_files:
    input_file_path = os.path.join(input_folder, file_name)
    output_file_path = os.path.join(output_folder, file_name)
    
    with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
        reader = csv.reader(input_file, delimiter=" ")
        for row in reader:
            cx, cy, cz, sx, sy, sz, yaw = [float(val) for val in row[:-1]]
            rot_y = yaw / 180.0 * pi
            theta = atan(cx / cy)
            kitti = KittiDescriptor()
            kitti.set_type('Car')
            kitti.set_occlusion(0.00)
            kitti.set_truncated(0) #前几项就按0和默认值写了
            kitti.set_alpha(rot_y - theta)
            #这个角是最抽象的，正常应该是用rot_y角减去theta角得到这个角的值 但是我没搞清楚
            #theta角到底在这个坐标系里应该是什么样的一个值，所以我就先按自己想象中写了，就是求个反正切值
            #可以参考https://blog.csdn.net/qq_16137569/article/details/118873033的图

            kitti.set_bbox([0,0,0,0]) #这个bbox好像只用点坐标求不出来
            kitti.set_3d_object_dimensions(sx, sy, sz) #这个按照网上描述的，高 宽 长的格式来写
            kitti.set_3d_object_location(cx, cy, cz - sz) 
            #底面中心 减去一个sz，但是我还是没完全搞懂到底应该用哪个坐标系
            kitti.set_rotation_y(rot_y) 
            #如果没错的话 这两个角应该都是与x轴的夹角，能直接用，但是问题是坐标系变了 我一时间也搞不清到底应该是哪个角
            #需要改成弧度制
            output_file.write(kitti.__str__())
            output_file.write('\n')
