import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import math

class Bev_Heatmap:
    def main(self):
        pcd_path, txt_path = self.config_path()
        pre_point = np.fromfile(str(pcd_path), dtype=np.dtype([
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('intensity', np.float32),
                                    ]) ,count=-1)

        pcd = np.array([list(elem) for elem in pre_point])
        subset_pcd = pcd
        subset_pcd = []
        sub_r = 50
        for point in pcd:
            if point[0] > -sub_r and point[0] < sub_r and point[1] > -sub_r and point[1] < sub_r:
                subset_pcd.append(point)
        

        t0 = time.time()

        delta_depth_map  = self.pillar_model(subset_pcd)
        
        t1 = time.time()
        print("time:",t1-t0)

        # self.visualize_heatmap(delta_depth_map)

        entropy = self.get_scene_entropy(subset_pcd)
        t2 = time.time()

        print("entropy:",entropy)
        print("time:",t2-t1)

        self.visualize_heatmap(delta_depth_map)

        return
    

    def config_path(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', type=str, default='1729')
        parser.add_argument('-s', type=str, default='0000404165')
        args = parser.parse_args()


        dir = args.d
        spec = args.s

        pcd = "/home/ghosnp/carla/usable_version_tool/uv2/carla_dataset_tools/raw_data/record_2023_"+dir+"/vehicle.tesla.model3_2/velodyne/"+spec+".bin"
        txt = "/home/ghosnp/carla/usable_version_tool/uv2/carla_dataset_tools/raw_data/record_2023_"+dir+"/vehicle.tesla.model3_2/velodyne_semantic/"+spec+".txt"

        return pcd,txt

    def pillar_model(self,points):
        # segment points into pillars
        r = 2
 
        max_range = np.max(points, axis=0)
        min_range = np.min(points, axis=0)
        min_x = min_range[0]
        max_x = max_range[0]
        min_y = min_range[1]
        max_y = max_range[1]
        min_z = min_range[2]
        max_z = max_range[2]

        # get the number of pillars
        self.count_x = int((max_x - min_x) / r)+1
        self.count_y = int((max_y - min_y) / r)+1

        # create a pillar model(x-y plane), and initialize it
        max_depth = np.full((self.count_x,self.count_y),-1000)
        min_depth = np.full((self.count_x,self.count_y),1000)


        # calculate the pillar model
        for point in points:
            if point[2] > max_depth[int((point[0] - min_x) / r)][int((point[1] - min_y) / r)]:
                max_depth[int((point[0] - min_x) / r)][int((point[1] - min_y) / r)] = point[2] 
            if point[2] < min_depth[int((point[0] - min_x) / r)][int((point[1] - min_y) / r)]:
                min_depth[int((point[0] - min_x) / r)][int((point[1] - min_y) / r)] = point[2] 
            
        delta_depth_map = np.zeros((self.count_x,self.count_y))

        # calculate the delta depth map
        for i in range(self.count_x):
            for j in range(self.count_y):
                if max_depth[i][j] != -1000 and min_depth[i][j] != 1000:
                    depth = max_depth[i][j] - min_depth[i][j]
                    delta_depth_map[i][j] = depth

        return delta_depth_map
    
    
    def visualize_heatmap(self,delta_depth_map):
        data = np.random.rand(10, 10)

        plt.imshow(delta_depth_map, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()

        return
    
    def get_scene_entropy(self, points):
        voxel_size = 2
        voxel_max_range = np.max(points, axis=0)
        voxel_min_range = np.min(points, axis=0)

        voxel_count = [int((voxel_max_range[0] - voxel_min_range[0])/voxel_size)+1,
                        int((voxel_max_range[1]-voxel_min_range[1])/voxel_size)+1,
                        int((voxel_max_range[2]-voxel_min_range[2])/voxel_size)+1]
        
        voxel_scene = np.zeros(voxel_count)
        for point in points:
            voxel_scene[int((point[0] - voxel_min_range[0])/voxel_size)][int((point[1] - voxel_min_range[1])/voxel_size)][int((point[2] - voxel_min_range[2])/voxel_size)] += 1

        dt = len(points) / ((voxel_max_range[0] - voxel_min_range[0]) * (voxel_max_range[1] - voxel_min_range[1]) * (voxel_max_range[2] - voxel_min_range[2]))
        entropy = 0
        for i in range(voxel_count[0]):
            for j in range(voxel_count[1]):
                for k in range(voxel_count[2]):
                    di = voxel_scene[i][j][k]
                    if di != 0:
                        entropy -= (di / dt) *  math.log10(di / dt)

        return entropy
    
    def cal_entropy_if_keep(self, entropy_last, entropy_now):
        return abs(entropy_now - entropy_last) / -entropy_last >= self._Hs

if __name__ == '__main__':
    heatmap = Bev_Heatmap()
    heatmap.main()