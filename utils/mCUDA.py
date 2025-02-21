import numpy as np

def transform_matrices(A, B):
    """
    将两个矩阵 A 和 B 转换为新的矩阵 A` 和 B`，使得 A` 的每行是 A 的一行的重复，B` 的每行是 B 的一行的重复
    用于计算两个矩阵中每对行之间的距离
    """
    n1, dim = A.shape
    n2, _ = B.shape

    # 将 A 逐行复制 n2 次
    # A_prime = np.repeat(A, n2, axis=0)
    A_prime = np.tile(A, (n2, 1))

    # 将 B 按行复制 n1 次
    # B_prime = np.tile(B, (n1, 1))
    B_prime = np.repeat(B, n1, axis=0)
    return A_prime, B_prime

# 示例
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

A_prime, B_prime = transform_matrices(A, B)
print("A`:\n", A_prime)
print("B`:\n", B_prime)

import psutil
import GPUtil
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA kernel for matrix multiplication
mod = SourceModule("""
__global__ void matMulKernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (row < M && col < K) {
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
""")

def measure_time_numpy(A_list, B):
    start_time = time.time()
    for A in A_list:
        C = np.dot(A, B)
    end_time = time.time()
    return end_time - start_time

def measure_time_pycuda(A_list, B):
    M, N = A_list[0].shape
    K = B.shape[1]
    block_size = 16
    grid_size_x = (K + block_size - 1) // block_size
    grid_size_y = (M + block_size - 1) // block_size

    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(A_list[0].shape[0] * B.shape[1] * np.float32().nbytes)
    func = mod.get_function("matMulKernel")

    start_time = time.time()
    cuda.memcpy_htod(B_gpu, B)
    for A in A_list:
        A_gpu = cuda.mem_alloc(A.nbytes)
        cuda.memcpy_htod(A_gpu, A)
        func(A_gpu, B_gpu, C_gpu, np.int32(M), np.int32(N), np.int32(K), block=(block_size, block_size, 1), grid=(grid_size_x, grid_size_y))
        cuda.Context.synchronize()
        A_gpu.free()
    end_time = time.time()
    return end_time - start_time

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    gpu_usage = []
    for gpu in gpus:
        gpu_usage.append((gpu.id, gpu.load * 100, gpu.memoryUtil * 100))
    return gpu_usage

# 生成示例数据
A_list = [np.random.rand(1521, 170).astype(np.float32) for _ in range(10000)]
B = np.random.randint(2, size=(170, 30)).astype(np.float32)

# 获取初始 CPU 和 GPU 使用情况
initial_cpu_usage = get_cpu_usage()
initial_gpu_usage = get_gpu_usage()

# 使用 NumPy 进行矩阵乘法
numpy_time = measure_time_numpy(A_list, B)
print(f"NumPy time: {numpy_time:.6f} seconds")

# 获取 NumPy 执行后的 CPU 和 GPU 使用情况
numpy_cpu_usage = get_cpu_usage()
numpy_gpu_usage = get_gpu_usage()

# 使用 PyCUDA 进行矩阵乘法
pycuda_time = measure_time_pycuda(A_list, B)
print(f"PyCUDA time: {pycuda_time:.6f} seconds")

# 获取 PyCUDA 执行后的 CPU 和 GPU 使用情况
pycuda_cpu_usage = get_cpu_usage()
pycuda_gpu_usage = get_gpu_usage()

# 打印资源使用情况
print(f"Initial CPU usage: {initial_cpu_usage}%")
print(f"Initial GPU usage: {initial_gpu_usage}")
print(f"NumPy CPU usage: {numpy_cpu_usage}%")
print(f"NumPy GPU usage: {numpy_gpu_usage}")
print(f"PyCUDA CPU usage: {pycuda_cpu_usage}%")
print(f"PyCUDA GPU usage: {pycuda_gpu_usage}")

# 评价为 PyCUDA 加速比在维度较低时是有限的，因为数据传输和内核启动的开销可能会超过计算本身的时间
# 所以直接使用numpy进行并行计算优化是更好的选择