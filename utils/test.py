import cupy as cp
import numpy as np
import time
class knn_classifier_for_static_data:
    def __init__(self,xtrain,ytrain,xtest,ytest,k):
        # 这里写一些维度检查的代码--暂无
        # 这里写初始化GPU数据的代码
        n1,dim = xtrain.shape
        n2,dim  = xtest.shape
        xtrain = cp.asarray(xtrain)
        print("xtrain shape:", xtrain.shape)
        xtest  = cp.asarray(xtest )
        xtrain = cp.broadcast_to(xtrain[cp.newaxis,:,:],(n2,n1,dim))
        xtest  = cp.broadcast_to(xtest[:,cp.newaxis,:],(n2,n1,dim))
        print("xtest shape:", xtest.shape)
        Dtrain_test = xtrain - xtest
        Dtrain_test = Dtrain_test ** 2
        print("D_train_test shape:", Dtrain_test.shape)
        ytrainT= ytrain.T
        YtrainT = cp.asarray(ytrainT)
        YtrainT= cp.broadcast_to(YtrainT[cp.newaxis,:,:],(n2,1,n1))
        Ytest  = cp.asarray(ytest)
        # 这里写存储的数据
        self.Dtrain_test = Dtrain_test
        self.YtrainT     = YtrainT
        self.Ytest       = Ytest
        self.n2          = n2
        self.n1          = n1
        self.k           = k
        
        # 打印变量的形状       
        print("y_train_T shape:", YtrainT.shape)
        print("ytest shape:", ytest.shape)
    def muti_classifier(self,populationList):
        # 这里写一些维度检查的代码--暂无
        # 这里写一些将数据放入GPU的代码
        the_population = populationList[0].shape[1]
        print('the population ',the_population)
        # 这一步是错的，需要将原本矩阵按列合并
        combined_array = np.concatenate(populationList,axis=1)
        print("合并后的 NumPy 数组形状:", combined_array.shape)
        dim,N          = combined_array.shape
        combined_array = cp.asarray(combined_array)
        combined_array = cp.broadcast_to(combined_array[cp.newaxis,:,:],(self.n2,dim,N))
        print("合并后的 NumPy 数组形状now:", combined_array.shape)
        Dtrain_test_now= cp.matmul(self.Dtrain_test,combined_array)
        print("Dtrain_test_now shape:", Dtrain_test_now.shape)
        # 排序-取0-1矩阵
        sorted_result  = cp.zeros_like(Dtrain_test_now)
        sorted_indices = cp.argsort(Dtrain_test_now,axis=1)
        print("sorted_result shape:", sorted_result.shape)
        print("sorted_indices shape:", sorted_indices.shape)
        for axis_val in range(self.n2):
            for n in range(N):
                min_indices = sorted_indices[axis_val,:self.k,n]
                sorted_result[axis_val,min_indices,n] = 1
        # 与Ytrain进行矩阵乘法
        yPred = cp.matmul(self.YtrainT,sorted_result)
        yPred = cp.round(yPred)
        YtestN= cp.broadcast_to(self.Ytest[:,:,cp.newaxis],(self.n2,1,N))
        error = cp.sum(cp.abs(yPred - YtestN),axis=0).reshape(N)/self.n2
        print("yPred shape:", yPred.shape)
        print("YtestN shape:", YtestN.shape)
        YtestN = YtestN.reshape(self.n2,N)
        print("error shape:", error.shape)
        return cp.asnumpy(error)
class np_knn_classifier_for_static_data:
    def __init__(self,xtrain,ytrain,xtest,ytest,k):
        # 这里写一些维度检查的代码--暂无
        # 这里写初始化GPU数据的代码
        n1,dim = xtrain.shape
        n2,dim  = xtest.shape
        # xtrain = cp.asarray(xtrain)
        print("xtrain shape:", xtrain.shape)
        # xtest  = cp.asarray(xtest )
        xtrain = np.broadcast_to(xtrain[np.newaxis,:,:],(n2,n1,dim))
        xtest  = np.broadcast_to(xtest[:,np.newaxis,:],(n2,n1,dim))
        print("xtest shape:", xtest.shape)
        Dtrain_test = xtrain - xtest
        Dtrain_test = Dtrain_test ** 2
        print("D_train_test shape:", Dtrain_test.shape)
        ytrainT= ytrain.T
        # YtrainT = cp.asarray(ytrainT)
        YtrainT= np.broadcast_to(ytrainT[np.newaxis,:,:],(n2,1,n1))
        Ytest  = np.asarray(ytest)
        # 这里写存储的数据
        self.Dtrain_test = Dtrain_test
        self.YtrainT     = YtrainT
        self.Ytest       = Ytest
        self.n2          = n2
        self.n1          = n1
        self.k           = k
        
        # 打印变量的形状       
        print("y_train_T shape:", YtrainT.shape)
        print("ytest shape:", ytest.shape)
    def muti_classifier(self,populationList):
        # 这里写一些维度检查的代码--暂无
        # 这里写一些将数据放入GPU的代码
        the_population = populationList[0].shape[1]
        print('the population ',the_population)
        # 这一步是错的，需要将原本矩阵按列合并
        combined_array = np.concatenate(populationList,axis=1)
        print("合并后的 NumPy 数组形状:", combined_array.shape)
        dim,N          = combined_array.shape
        # combined_array = cp.asarray(combined_array)
        combined_array = np.broadcast_to(combined_array[np.newaxis,:,:],(self.n2,dim,N))
        print("合并后的 NumPy 数组形状now:", combined_array.shape)
        Dtrain_test_now= np.matmul(self.Dtrain_test,combined_array)
        print("Dtrain_test_now shape:", Dtrain_test_now.shape)
        # 排序-取0-1矩阵
        sorted_result  = np.zeros_like(Dtrain_test_now)
        sorted_indices = np.argsort(Dtrain_test_now,axis=1)
        print("sorted_result shape:", sorted_result.shape)
        print("sorted_indices shape:", sorted_indices.shape)
        for axis_val in range(self.n2):
            for n in range(N):
                min_indices = sorted_indices[axis_val,:self.k,n]
                sorted_result[axis_val,min_indices,n] = 1
        # 与Ytrain进行矩阵乘法
        yPred = np.matmul(self.YtrainT,sorted_result)
        yPred = np.round(yPred)
        YtestN= np.broadcast_to(self.Ytest[:,:,np.newaxis],(self.n2,1,N))
        error = np.sum(np.abs(yPred - YtestN),axis=0).reshape(N)/self.n2
        print("yPred shape:", yPred.shape)
        print("YtestN shape:", YtestN.shape)
        YtestN = YtestN.reshape(self.n2,N)
        print("error shape:", error.shape)
        return error
if __name__ == '__main__':
    # 现在使用一些测试用例测试对象
    # 创建一个3x4的随机浮点矩阵
    np.random.seed(42)
    n1         = 180
    n2         = 20
    dim        = 170
    matrix_3x4 = np.random.rand(n1, dim)
    matrix_4x4 = np.random.rand(n2,dim)
    # 创建一个3x1的随机0-1矩阵
    matrix_3x1 = np.random.randint(0, 2, size=(n1, 1))
    matrix_4x1 =np.random.randint(0,2,size=(n2,1))
    newKnnClassifier =knn_classifier_for_static_data(matrix_4x4,matrix_4x1,matrix_3x4,matrix_3x1,2)
    newKnnClassifier_np = np_knn_classifier_for_static_data(matrix_4x4,matrix_4x1,matrix_3x4,matrix_3x1,2)
    # 创建一个长度为10的随机4x2列表
    the_choice = [10,100,1000]
    for i in range(len(the_choice)):        
        len_of_array = the_choice[i]
        list_4x2 = [np.random.rand(dim, 30) for _ in range(len_of_array)]
        st = time.time()
        error = newKnnClassifier.muti_classifier(list_4x2)
        et = time.time()
        print('time used in GPU', et - st)
        # print(error)
        error = newKnnClassifier_np.muti_classifier(list_4x2)
        print('time used CPU ',time.time()-et)

# 数据名称	形状	时间 (秒)	人口数量
# xtrain	(20, 170)	-	-
# xtest	(180, 20, 170)	-	-
# D_train_test	(180, 20, 170)	-	-
# y_train_T	(180, 1, 20)	-	-
# ytest	(180, 1)	-	-
# 合并后的 NumPy 数组	(170, 300)	-	30
# 合并后的 NumPy 数组now	(180, 170, 300)	-	30
# Dtrain_test_now	(180, 20, 300)	-	30
# sorted_result	(180, 20, 300)	-	30
# sorted_indices	(180, 20, 300)	-	30
# yPred	(180, 1, 300)	-	30
# YtestN	(180, 1, 300)	-	30
# error	(300,)	-	30
# 时间统计	GPU: 3.715	CPU: 0.120	人口: 30
# 合并后的 NumPy 数组	(170, 3000)	-	30
# 合并后的 NumPy 数组now	(180, 170, 3000)	-	30
# Dtrain_test_now	(180, 20, 3000)	-	30
# sorted_result	(180, 20, 3000)	-	30
# sorted_indices	(180, 20, 3000)	-	30
# yPred	(180, 1, 3000)	-	30
# YtestN	(180, 1, 3000)	-	30
# error	(3000,)	-	30
# 时间统计	GPU: 34.760	CPU: 1.140	人口: 30
# 合并后的 NumPy 数组	(170, 30000)	-	30
# 合并后的 NumPy 数组now	(180, 170, 30000)	-	30
# Dtrain_test_now	(180, 20, 30000)	-	30
# sorted_result	(180, 20, 30000)	-	30
# sorted_indices	(180, 20, 30000)	-	30
# yPred	(180, 1, 30000)	-	30
# YtestN	(180, 1, 30000)	-	30
# error	(30000,)	-	30
# 时间统计	GPU: 539.532	CPU: 11.346	人口: 30
# 结论：不如不使用GPU加速Emma