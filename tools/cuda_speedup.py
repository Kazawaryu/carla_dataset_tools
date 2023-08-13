import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Define our function using CUDA code
cuda_func_def = """
__global__ void multiply(float *result, float *a, float *b)
{
  const int i = threadIdx.x;
  result[i] = a[i] * b[i];
}
"""
        
# Create CUDA module and import into Python
mod = SourceModule(cuda_func_def)
multiply_func = mod.get_function("multiply")

# create Python variables
a = np.random.randn(100).astype(np.float32)	 
b = np.random.randn(100).astype(np.float32)
result = np.random.randn(100).astype(np.float32)

# allocate memory on GPU
a_gpu = cuda.mem_alloc(a.nbytes)		
b_gpu = cuda.mem_alloc(b.nbytes)
result_gpu = cuda.mem_alloc(b.nbytes)

# copy data to GPU
cuda.memcpy_htod(a_gpu, a)		
cuda.memcpy_htod(b_gpu, b)

# run multiply function on GPU
multiply_func(    			
    result_gpu, a_gpu, b_gpu,
    block=(100, 1, 1), 
    grid=(1, 1)
)

# Get data back from GPU
cuda.memcpy_dtoh(result, result_gpu)
print(result)	

assert np.allclose(result, np.multiply(a,b))