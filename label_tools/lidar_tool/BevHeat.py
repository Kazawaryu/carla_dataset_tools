import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import math

import numba

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

@numba.jit(nopython=True)
def calculate_depth(points, min_x, min_y, inv_r, max_depth, min_depth):
    # for point in points:
    #     x_index = int((point[0] - min_x) / r)
    #     y_index = int((point[1] - min_y) / r)
    #     if point[2] > max_depth[x_index][y_index]:
    #         max_depth[x_index][y_index] = point[2]
    #     if point[2] < min_depth[x_index][y_index]:
    #         min_depth[x_index][y_index] = point[2]
    # return max_depth, min_depth
        

        for i, point in enumerate(points):
            x = point[0]
            y = point[1]
            z = point[2]
            x_index = int((x - min_x) * inv_r)
            y_index = int((y - min_y) * inv_r)

            if z > max_depth[x_index][y_index]:
                max_depth[x_index][y_index] = z
            if z < min_depth[x_index][y_index]:
                min_depth[x_index][y_index] = z

        return max_depth, min_depth

@numba.jit(nopython=True) 
def calculate_delta_depth_map(count_x, count_y, max_depth, min_depth):
    delta_depth_map = np.zeros((count_x, count_y))
    for i in range(count_x):
        for j in range(count_y):
            if max_depth[i][j] != -1000 and min_depth[i][j] != 1000:
                depth = max_depth[i][j] - min_depth[i][j]
                delta_depth_map[i][j] = depth
    return delta_depth_map



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

        # entropy = self.get_scene_entropy(subset_pcd)
        # entropy = self.cuda_speedup_get_scene_entropy(subset_pcd)
        entropy = self.numpy_speedup_get_scene_entropy(subset_pcd)
        t2 = time.time()

        print("entropy:",entropy)
        print("time:",t2-t1)

        print("now test speedup algorithm")
        for i in range(1):
            t2 = time.time()
            self.numba_speedup_pillar_model(subset_pcd)
            # self.pillar_model(subset_pcd)
            t3 = time.time()
            print("idx: ",i,",numba:",t3-t2)




    def config_path(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', type=str, default='0805_1536')
        parser.add_argument('-s', type=str, default='0000000419')
        args = parser.parse_args()


        dir = args.d
        spec = args.s

        pcd = "/home/ghosnp/carla/usable_version_tool/uv2/carla_dataset_tools/raw_data/record_2023_"+dir+"/vehicle.tesla.model3_2/velodyne/"+spec+".bin"
        txt = "/home/ghosnp/carla/usable_version_tool/uv2/carla_dataset_tools/raw_data/record_2023_"+dir+"/vehicle.tesla.model3_2/velodyne_semantic/"+spec+".txt"

        return pcd,txt
    
    # ##############################################################################
    

    def numba_speedup_pillar_model(self,points):
        # segment points into pillars
        r = 2

        max_range = np.max(points, axis=0)
        min_range = np.min(points, axis=0)
        min_x = min_range[0]
        max_x = max_range[0]
        min_y = min_range[1]
        max_y = max_range[1]

 

        # get the number of pillars
        self.count_x = int((max_x - min_x) / r)+1
        self.count_y = int((max_y - min_y) / r)+1

        # create a pillar model(x-y plane), and initialize it
        max_depth = np.full((self.count_x,self.count_y),-1000)
        min_depth = np.full((self.count_x,self.count_y),1000)

        tt = time.time()
        # calculate the pillar model
        max_depth, min_depth = calculate_depth(points, min_x, min_y, 1/r, max_depth, min_depth)
        print("numba time_1:",time.time()-tt)

        
        # calculate the delta depth map
        delta_depth_map = calculate_delta_depth_map(self.count_x, self.count_y, max_depth, min_depth)
    

        return delta_depth_map

    # ##############################################################################

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
    
    def cuda_speedup_get_scene_entropy(self, points):
        cuda_func_def = """
        __global__ void calculate_voxel_frequency(int *voxel_scene, float *points, int voxel_size, int count_x, int count_y, int count_z, float min_x, float min_y, float min_z)
        {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            voxel_scene[(int)((points[i * 3] - min_x) / voxel_size) * count_y * count_z + (int)((points[i * 3+1] - min_y) / voxel_size) * count_z + (int)((points[i * 3+2] - min_z) / voxel_size)] += 1;
        }
        """
        voxel_size = 2
        voxel_max_range = np.max(points, axis=0)
        voxel_min_range = np.min(points, axis=0)

        voxel_count = [int((voxel_max_range[0] - voxel_min_range[0])/voxel_size),
                      int((voxel_max_range[1]-voxel_min_range[1])/voxel_size),
                      int((voxel_max_range[2]-voxel_min_range[2])/voxel_size)]
        voxel_scene = np.zeros(voxel_count)


        mod = SourceModule(cuda_func_def)
        multiply_them = mod.get_function("calculate_voxel_frequency")

        points = np.array(points, dtype=float)
        voxel_scene_2gpu = drv.mem_alloc(voxel_scene.nbytes)
        points_2gpu = drv.mem_alloc(points.nbytes)

        drv.memcpy_htod(voxel_scene_2gpu, voxel_scene)
        drv.memcpy_htod(points_2gpu, points)

        multiply_them(
            drv.Out(voxel_scene_2gpu), drv.In(points_2gpu), np.int32(voxel_size), np.int32(voxel_count[0]), np.int32(voxel_count[1]), np.int32(voxel_count[2]), np.float32(voxel_min_range[0]), np.float32(voxel_min_range[1]), np.float32(voxel_min_range[2]),
            block=(400,1,1), grid=(1,1))
        
        drv.memcpy_dtoh(voxel_scene, voxel_scene_2gpu)

        return voxel_scene


    

    # def get_scene_entropy(self, points):
    #     self.voxel_size = 2
    #     voxel_max_range = np.max(points, axis=0)
    #     voxel_min_range = np.min(points, axis=0)

    #     voxel_count = [int((voxel_max_range[0] - voxel_min_range[0])/self.voxel_size)+1,
    #                    int((voxel_max_range[1] - voxel_min_range[1])/self.voxel_size)+1,
    #                    int((voxel_max_range[2] - voxel_min_range[2])/self.voxel_size)+1]

    #     voxel_scene = np.zeros(voxel_count, dtype=np.uint32)
    #     points = np.array(points, dtype=np.float32)

    #     # Copy data to GPU
    #     points_gpu = cuda.to_device(points)
    #     voxel_scene_gpu = cuda.to_device(voxel_scene)

    #     # Define CUDA kernel
    #     mod = SourceModule("""
    #         #define VOXEL_SIZE %(voxel_size)d

    #         __global__ void calculate_voxel_scene(float* points, unsigned int* voxel_scene, float min_x, float min_y, float min_z, int count_x, int count_y, int count_z)
    #         {
    #             int idx = threadIdx.x + blockIdx.x * blockDim.x;
    #             int stride_x = blockDim.x * gridDim.x;

    #             for (int i = idx; i < points.shape[0]; i += stride_x)
    #             {
    #                 float x = points[i * 3];
    #                 float y = points[i * 3 + 1];
    #                 float z = points[i * 3 + 2];
    #                 int voxel_x = (int)((x - min_x) / VOXEL_SIZE);
    #                 int voxel_y = (int)((y - min_y) / VOXEL_SIZE);
    #                 int voxel_z = (int)((z - min_z) / VOXEL_SIZE);

                    
    #                 int index = voxel_x * count_y * count_z + voxel_y * count_z + voxel_z;
    #                 atomicAdd(&voxel_scene[index], 1);
                    
    #             }
    #         }
    #     """ % {'voxel_size': self.voxel_size})

    #     # Launch CUDA kernel
    #     block_size = 128
    #     grid_size = (points.shape[0] + block_size - 1) // block_size
    #     func = mod.get_function("calculate_voxel_scene")
    #     func(points_gpu, voxel_scene_gpu, np.float32(voxel_min_range[0]), np.float32(voxel_min_range[1]), np.float32(voxel_min_range[2]), 
    #          np.int32(voxel_count[0]), np.int32(voxel_count[1]), np.int32(voxel_count[2]), block=(block_size, 1, 1), grid=(grid_size, 1))

    #     # Copy result back to CPU
    #     voxel_scene = voxel_scene_gpu.get()

    #     dt = len(points) / ((voxel_max_range[0] - voxel_min_range[0]) * (voxel_max_range[1] - voxel_min_range[1]) * (voxel_max_range[2] - voxel_min_range[2]))
    #     entropy = 0
    #     for i in range(voxel_count[0]):
    #         for j in range(voxel_count[1]):
    #             for k in range(voxel_count[2]):
    #                 di = voxel_scene[i][j][k]
    #                 if di != 0:
    #                     entropy -= (di / dt) * np.log10(di / dt)

    #     return entropy
    
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
    

    def numpy_speedup_get_scene_entropy(self,points):
        voxel_size = 2
        voxel_max_range = np.max(points, axis=0)
        voxel_min_range = np.min(points, axis=0)

        voxel_count = np.ceil((voxel_max_range - voxel_min_range) / voxel_size).astype(int)[:3]
        voxel_scene = np.zeros(voxel_count)
        
        indices = np.floor((points - voxel_min_range) / voxel_size).astype(int)
        for i in indices:
            voxel_scene[i[0],i[1],i[2]] += 1

        dt = len(points) / ((voxel_max_range[0] - voxel_min_range[0]) * (voxel_max_range[1] - voxel_min_range[1]) * (voxel_max_range[2] - voxel_min_range[2]))

        nonzero_indices = np.nonzero(voxel_scene)
        di = voxel_scene[nonzero_indices]
        entropy = -np.sum((di / dt) * np.log10(di / dt))


        return entropy
    
    def cal_entropy_if_keep(self, entropy_last, entropy_now):
        return abs(entropy_now - entropy_last) / -entropy_last >= self._Hs

if __name__ == '__main__':
    heatmap = Bev_Heatmap()
    heatmap.main()


