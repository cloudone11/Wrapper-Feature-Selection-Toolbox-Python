import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time

# 定义矩阵类
class Matrix:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.elements = np.zeros((width, height), dtype=np.float32)

# CUDA kernel
mod = SourceModule("""
__global__ void matMulKernel(Matrix A, Matrix B, Matrix C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < A.width && y < A.height) {
        float sum = 0.0;
        for (int i = 0; i < A.width; ++i) {
            sum += A.elements[y * A.width + i] * B.elements[i * B.width + x];
        }
        C.elements[y * C.width + x] = sum;
    }
}

__global__ void initMatrix(float *elements, float value, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        elements[idx] = value;
    }
}
""")

# 定义kernel函数
matMulKernel = mod.get_function("matMulKernel")
initMatrix = mod.get_function("initMatrix")

# 初始化矩阵
width = 1 << 10
height = 1 << 10
A = Matrix(width, height)
B = Matrix(width, height)
C = Matrix(width, height)

# 分配GPU内存
A_gpu = drv.mem_alloc(A.elements.nbytes)
B_gpu = drv.mem_alloc(B.elements.nbytes)
C_gpu = drv.mem_alloc(C.elements.nbytes)

# 初始化数据
initMatrix(drv.In(A_gpu), np.float32(1.0), np.int32(width), np.int32(height), block=(256, 1, 1), grid=((width*height+255)//256, 1))
initMatrix(drv.In(B_gpu), np.float32(2.0), np.int32(width), np.int32(height), block=(256, 1, 1), grid=((width*height+255)//256, 1))

# 定义kernel的执行配置
block_size = (32, 32, 1)
grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

# 监控性能
start_time = time.time()

# 执行kernel
matMulKernel(drv.In(A_gpu), drv.In(B_gpu), drv.In(C_gpu), block=block_size, grid=grid_size)

# 同步device 保证结果能正确访问
drv.Context.synchronize()

# 检查执行结果
C.elements = np.empty_like(C.elements)
drv.memcpy_dtoh(C.elements, C_gpu)
max_error = np.max(np.abs(C.elements - 2 * width))

end_time = time.time()
print(f"最大误差: {max_error}")
print(f"执行时间: {end_time - start_time} 秒")