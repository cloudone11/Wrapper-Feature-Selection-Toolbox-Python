import numpy as np
import operator

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


mod = SourceModule("""

    __global__ void squareDistance(float* x1, float* y1, float* x2, float* y2, float* distence, int N)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < N; i += stride)
        {
            //printf("x1, y1: %f, 5%f", x1[i], y1[i]);
            distence[i] = (x2[0]-x1[i]) * (x2[0]-x1[i]) + (y2[0]-y1[i]) * (y2[0]-y1[i]);
        }
    }

""")


class KNN(object):

    def __init__(self, k=3):
        self.k = k


    def fit(self, x, y):
        self.x = x
        self.y = y
    

    def _vote(self, ys):
        ys_unique = np.unique(ys)
        # print('ys_unique, ys: %d %d'%(len(ys_unique), len(ys)))
        vote_dict = {}
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        """
            1. sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方
            法返回的是一个新的 list，而不是在原来的基础上进行的操作。
            2. operator 模块提供了一套与Python的内置运算符对应的高效率函数。
            例如，operator.add(x, y) 与表达式 x+y 相同。operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号），下面看例子。
            a = [1,2,3] 
            >>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
            >>> b(a) 
            2 
            >>> b=operator.itemgetter(1,0)   //定义函数b，获取对象的第1个域和第0个的值
            >>> b(a) 
            (2, 1) 
            要注意，operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。
        """
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_vote_dict[0][0])
        return sorted_vote_dict[0][0]


    def predict(self, x):
        square_distance = mod.get_function('squareDistance')
        y_pred = []
        
        n_size = len(self.x)
        n_size = np.int32(n_size)
        x1 = np.empty(n_size, dtype=np.float32)
        x1 = self.x[:,0]
        y1 = np.empty(n_size, dtype=np.float32)
        y1 = self.x[:,1]
        # print(self.x.shape)
        # 把x1, y1的排列方式转成行优先, 没有这一步会出错
        x1 = x1.copy(order='C')
        y1 = y1.copy(order='C')
        # 显卡原因因为是游戏显卡, 所以只能计算32位的浮点数
        # 注意: 申请x1的时候, 虽然写了x1的类型是float32,但还是要写下面这两句
        x1 = np.float32(x1)
        y1 = np.float32(y1)
        for i in range(len(x)):
            x2 = np.empty(1, dtype=np.float32)
            x2[0] = x[i][0]
            y2 = np.empty(1, dtype=np.float32)
            y2[0] = x[i][1]
            dist_arr = np.empty(n_size, dtype=np.float32)

            thread_size = 256
            block_size = int((n_size + thread_size - 1) / thread_size)

            square_distance(
                cuda.In(x1), cuda.In(y1), cuda.In(x2), cuda.In(y2), cuda.Out(dist_arr), n_size,
                block=(thread_size, 1, 1), grid=(block_size, 1)
            )
            sorted_index = np.argsort(dist_arr)
            top_k_index = sorted_index[:self.k]
            y_pred.append(self._vote(ys=self.y[top_k_index]))
        return np.array(y_pred)

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score = 0.0
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                score += 1
        score /= len(y_true)
        return score
