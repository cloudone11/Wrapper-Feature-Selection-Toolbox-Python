import cupy as cp
import numpy as np
import time
# from FS.functionHO import error_rate
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
# 结论：不如不使用GPU加速Emma,直接使用np进行矩阵乘法作0-1分类
def is_strict_binary_matrix(matrix):
        # 检查矩阵中的所有元素是否严格为0或1
        return np.all(np.isin(matrix, [0, 1]))
    
class knn_classifier_for_static_data:
    def __init__(self,xtrain,ytrain,xtest,ytest,k):
        # 这里写一些维度检查的代码--暂无
        if is_strict_binary_matrix(ytest) and is_strict_binary_matrix(ytrain):
            #print('the y and ytrain passed')
            1
        else:
            #print('the y and ytrain did not passed')
            raise Exception('Input matrices ytest and ytrain are not strict binary matrices')
        # 这里写初始化GPU数据的代码
        n1,dim = xtrain.shape
        n2,dim  = xtest.shape
        xtrain = cp.asarray(xtrain)
        #print("xtrain shape:", xtrain.shape)
        xtest  = cp.asarray(xtest )
        xtrain = cp.broadcast_to(xtrain[cp.newaxis,:,:],(n2,n1,dim))
        xtest  = cp.broadcast_to(xtest[:,cp.newaxis,:],(n2,n1,dim))
        #print("xtest shape:", xtest.shape)
        Dtrain_test = xtrain - xtest
        Dtrain_test = Dtrain_test ** 2
        #print("D_train_test shape:", Dtrain_test.shape)
        # ytrainT= ytrain.T
        # YtrainT = cp.asarray(ytrainT)
        ytrain = cp.asarray(ytrain)
        YtrainT= cp.broadcast_to(ytrain[cp.newaxis,:,:],(n2,n1,1))
        Ytest  = cp.asarray(ytest)
        # 这里写存储的数据
        self.Dtrain_test = Dtrain_test
        self.YtrainT     = YtrainT
        self.Ytest       = Ytest
        self.n2          = n2
        self.n1          = n1
        self.k           = k
        
        # 打印变量的形状       
        #print("y_train_T shape:", YtrainT.shape)
        #print("ytest shape:", ytest.shape)
    def muti_classifier(self,populationList):
        # 这里写一些维度检查的代码--暂无
        # 这里写一些将数据放入GPU的代码
        the_population = populationList[0].shape[1]
        #print('the population ',the_population)
        # 这一步是错的，需要将原本矩阵按列合并
        combined_array = np.concatenate(populationList,axis=1)
        #print("合并后的 NumPy 数组形状:", combined_array.shape)
        if is_strict_binary_matrix(combined_array):
            #print('pass check')
            1
        else:
            #print('failed to pass')
            raise Exception('Input matrices the_population are not strict binary matrices')
        
        dim,N          = combined_array.shape
        combined_array = cp.asarray(combined_array)
        combined_array = cp.broadcast_to(combined_array[cp.newaxis,:,:],(self.n2,dim,N))
        #print("合并后的 CuPy 数组形状now:", combined_array.shape)
         
        Dtrain_test_now= cp.matmul(self.Dtrain_test,combined_array)
        #print("Dtrain_test_now shape:", Dtrain_test_now.shape)
        # 开平方
        Dtrain_test_now= cp.sqrt(Dtrain_test_now)
        # 排序-取0-1矩阵
        sorted_result  = cp.zeros_like(Dtrain_test_now)
        sorted_indices = cp.argsort(Dtrain_test_now,axis=1)
        #print("sorted_result shape:", sorted_result.shape)
        #print("sorted_indices shape:", sorted_indices.shape)
        for axis_val in range(self.n2):
            for n in range(N):
                min_indices = sorted_indices[axis_val,:self.k,n]
                sorted_result[axis_val,min_indices,n] = 1
        # 与Ytrain进行矩阵乘法
        sorted_result = sorted_result.transpose(0,2,1)
        #print("sorted_result shape:", sorted_result.shape)
        #print("y_train_T shape:", self.YtrainT.shape)
        yPred = cp.matmul(sorted_result,self.YtrainT)
        yPred = cp.round(yPred/self.k)
        YtestN= cp.broadcast_to(self.Ytest[:,cp.newaxis,:],(self.n2,N,1))
        error = cp.sum(cp.abs(yPred - YtestN),axis=0).reshape(N)/self.n2
        #print("yPred shape:", yPred.shape)
        #print("YtestN shape:", YtestN.shape)
        YtestN = YtestN.reshape(self.n2,N)
        #print("error shape:", error.shape)
        return cp.asnumpy(error)
class np_knn_classifier_for_static_data:
    def __init__(self,xtrain,ytrain,xtest,ytest,k):
        if is_strict_binary_matrix(ytest) and is_strict_binary_matrix(ytrain):
            #print('the y and ytrain passed')
            1
        else:
            #print('the y and ytrain did not passed')
            raise Exception('Input matrices ytest and ytrain are not strict binary matrices')
        ytrain = ytrain.reshape(-1,1)
        ytest  = ytest.reshape(-1,1)
        # 这里写初始化输入数据类型的代码
        xtrain,xtest = np.asarray(xtrain,dtype='float'),np.asarray(xtest,dtype='float')
        ytrain,ytest = np.asarray(ytrain,dtype='int'),np.asarray(ytest,dtype='int')
        # 这里写初始化GPU数据的代码
        n1,dim = xtrain.shape
        n2,dim1  = xtest.shape
        if dim1 != dim or ytrain.shape[0]!= n1 or ytest.shape[0]!=n2 :
            print('the input is invalid')
            raise Exception('input error in dimension')
        # xtrain = cp.asarray(xtrain)
        #print("xtrain shape:", xtrain.shape)
        # xtest  = cp.asarray(xtest )
        xtrain = np.broadcast_to(xtrain[np.newaxis,:,:],(n2,n1,dim))
        xtest  = np.broadcast_to(xtest[:,np.newaxis,:],(n2,n1,dim))
        #print("xtest shape:", xtest.shape)
        Dtrain_test = xtrain - xtest
        Dtrain_test = Dtrain_test ** 2
        #print("D_train_test shape:", Dtrain_test.shape)
        # ytrainT= ytrain.T
        # YtrainT = cp.asarray(ytrainT)
        YtrainT= np.broadcast_to(ytrain[np.newaxis,:,:],(n2,n1,1))
        Ytest  = ytest
        # 这里写存储的数据
        self.Dtrain_test = Dtrain_test
        self.YtrainT     = YtrainT
        self.Ytest       = Ytest
        self.n2          = n2
        self.n1          = n1
        self.k           = k
        
        # 打印变量的形状       
        #print("y_train_T shape:", YtrainT.shape)
        #print("ytest shape:", ytest.shape)
    def muti_classifier(self,populationList):
        # 这里写一些维度检查的代码--暂无
        # 这里写一些将数据放入GPU的代码
        the_population = populationList[0].shape[0]
        #print('the population ',the_population)
        # 这一步是错的，需要将原本矩阵按列合并
        combined_array = np.concatenate(populationList,axis=0)
        #print("合并后的 NumPy 数组形状:", combined_array.shape)
        combined_array = combined_array.T
        #print('转置后的矩阵形状')
        if is_strict_binary_matrix(combined_array):
            #print('pass check')
            1
        else:
            #print('failed to pass')
            raise Exception('Input matrices the_population are not strict binary matrices') 
        dim,N          = combined_array.shape
        # combined_array = cp.asarray(combined_array)
        combined_array = np.broadcast_to(combined_array[np.newaxis,:,:],(self.n2,dim,N))
        #print("合并后的 NumPy 数组形状now:", combined_array.shape)
        Dtrain_test_now= np.matmul(self.Dtrain_test,combined_array)
        #print("Dtrain_test_now shape:", Dtrain_test_now.shape)
        # 开平方
        Dtrain_test_now= np.sqrt(Dtrain_test_now)
        # 排序-取0-1矩阵
        sorted_result  = np.zeros_like(Dtrain_test_now)
        sorted_indices = np.argsort(Dtrain_test_now,axis=1)
        #print("sorted_result shape:", sorted_result.shape)
        #print("sorted_indices shape:", sorted_indices.shape)
        for axis_val in range(self.n2):
            for n in range(N):
                min_indices = sorted_indices[axis_val,:self.k,n]
                sorted_result[axis_val,min_indices,n] = 1
        # 与Ytrain进行矩阵乘法
        sorted_result = sorted_result.transpose(0,2,1)
        #print("sorted_result shape:", sorted_result.shape)
        #print("y_train_T shape:", self.YtrainT.shape)
        yPred = np.matmul(sorted_result,self.YtrainT)
        yPred = np.round(yPred/self.k)
        YtestN= np.broadcast_to(self.Ytest[:,np.newaxis,:],(self.n2,N,1))
        error = np.sum(np.abs(yPred - YtestN),axis=0).reshape(N)/self.n2
        #print("yPred shape:", yPred.shape)
        #print("YtestN shape:", YtestN.shape)
        YtestN = YtestN.reshape(self.n2,N)
        #print("error shape:", error.shape)
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
    newKnnClassifier =knn_classifier_for_static_data(matrix_3x4,matrix_3x1,matrix_4x4,matrix_4x1,5)
    newKnnClassifier_np = np_knn_classifier_for_static_data(matrix_3x4,matrix_3x1,matrix_4x4,matrix_4x1,5)
    # 创建一个长度为10的随机4x2列表
    the_choice = [1]
    for i in range(len(the_choice)):        
        len_of_array = the_choice[i]
        list_4x2 = [np.random.randint(0,2,size=(dim,30)) for _ in range(len_of_array)]
        st = time.time()
        error1 = newKnnClassifier.muti_classifier(list_4x2)
        et = time.time()
        #print('time used in GPU', et - st)
        # #print(error)
        error2 = newKnnClassifier_np.muti_classifier(list_4x2)
        #print('time used CPU ',time.time()-et)
        #print('error 1 : \n',error1)
        #print('error 2 : \n',error2)
    for i in range(len(the_choice)):    
        error3 = np.zeros(30*the_choice[i])
        k = 5
        opts     = {}
        opts['k']= 5
        fold = {}
        opts['fold']=fold
        fold['xt']  = matrix_3x4
        fold['yt']  = matrix_3x1.reshape(-1)
        fold['xv']  = matrix_4x4
        fold['yv']  = matrix_4x1.reshape(-1)
        et = time.time()
        for i in range(error3.shape[0]):
            error3[i] = error_rate(matrix_4x4,matrix_4x1,np.round(list_4x2[i//30][:,i%30]),opts=opts)
        #print("for native knn classifier, used time: ", time.time()-et)    
        
        #print('error 3 : \n',error3)
        
    import numpy as np

    # 训练数据
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])

    # 测试数据
    X_test = np.array([[2.5, 3.5], [3.5, 4.5]])

    # 计算欧几里得距离
    def euclidean_distance(X_test, X_train):
        return np.sqrt(np.sum((X_test[:, np.newaxis] - X_train)**2, axis=2))

    # 找到最近的邻居
    def find_nearest_neighbors(distances, k):
        return np.argsort(distances, axis=1)[:, :k]

    # 进行投票
    def vote(neighbors, y_train):
        return np.array([np.argmax(np.bincount(y_train[neighbor])) for neighbor in neighbors])

    # k-NN 分类器
    def knn(X_train, y_train, X_test, k=3):
        distances = euclidean_distance(X_test, X_train)
        neighbors = find_nearest_neighbors(distances, k)
        predictions = vote(neighbors, y_train)
        return predictions

    # 使用分类器
    predictions = knn(X_train, y_train, X_test, k=3)
    #print("预测标签:", predictions)
    from sklearn.neighbors import KNeighborsClassifier
    mdl = KNeighborsClassifier(n_neighbors=3)
    mdl.fit(X_train, y_train)
    ypred = mdl.predict(X_test)
    #print("预测标签:", ypred)
    import numpy as np

    # 创建一个NumPy数组
    matrix = np.array([[4, 9], [16, 25]])

    # 对数组进行逐元素开方
    sqrt_matrix = np.sqrt(matrix)

    #print(sqrt_matrix)
    

    matrix = np.array([[0, 1, 1],
                [1, 0, 1],
                [0, 0, 0]])

    result = is_strict_binary_matrix(matrix)
    #print("Is the matrix a strict binary matrix?", result)