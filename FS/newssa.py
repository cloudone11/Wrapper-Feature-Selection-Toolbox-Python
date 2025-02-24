# Research and Application of A Novel Swarm Intelligence
# Optimization Technique: Sparrow Search Algorithm

# 麻雀搜索算法
# 文摘阅读 433
# 下载 516
# 导出题录 86
# 被引 502
# https://d.wanfangdata.com.cn/thesis/D02033402
# 值得一试
# -*- coding: utf-8 -*-
""" 
@Time    : 2022/10/20 22:39
@Author  : Xu Yong Kang 
"""

import numpy as  np

import numpy as np
import copy
import random
def initialization(pop,ub,lb,dim):
    ''' 种群初始化函数'''
    '''
    pop:为种群数量
    dim:每个个体的维度
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop,dim]) #声明空间
    for i in range(pop):
        for j in range(dim):
            X[i,j]=(ub[j]-lb[j])*np.random.random()+lb[j] #生成[lb,ub]之间的随机数
    
    return X
     
def BorderCheck(X,ub,lb,pop,dim):
    '''边界检查函数'''
    '''
    dim:为每个个体数据的维度大小
    X:为输入数据，维度为[pop,dim]
    ub:为个体数据上边界，维度为[dim,1]
    lb:为个体数据下边界，维度为[dim,1]
    pop:为种群数量
    '''
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X
 
 
def CaculateFitness(X,fun):
    '''计算种群的所有个体的适应度值'''
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness
 
 
def SortFitness(Fit):
    '''适应度值排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index
 
def SortPosition(X,index):
    '''根据适应度值对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew
 
 
def SSA(pop,dim,lb,ub,Max_iter,fun):
    '''麻雀搜索算法'''
    '''
    输入：
    pop:为种群数量
    dim:每个个体的维度
    ub:为个体上边界信息，维度为[1,dim]
    lb:为个体下边界信息，维度为[1,dim]
    fun:为适应度函数接口
    MaxIter:为最大迭代次数
    输出：
    GbestScore:最优解对应的适应度值
    GbestPositon:最优解
    Curve:迭代曲线
    '''
    
    ST = 0.8 #预警值
    PD = 0.2 #发现者的比列，剩下的是加入者
    SD = 0.1 #意识到有危险麻雀的比重
    PDNumber = int(pop*PD) #发现者数量
    SDNumber = int(pop*SD) #意识到有危险麻雀数量
    X = initialization(pop,ub,lb,dim) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])
    Curve = np.zeros([Max_iter,1])
    for t in range(Max_iter):
        print("第"+str(t)+"次迭代")
        BestF = copy.copy(fitness[0])
        Xworst = copy.copy(X[-1,:])
        Xbest = copy.copy(X[0,:])
        '''发现者位置更新'''
        R2 = np.random.random()
        for i in range(PDNumber):
            if R2<ST:
                X[i,:] = X[i,:]*np.exp(-i/(np.random.random()*Max_iter))
            else:
                X[i,:] = X[i,:] + np.random.randn()*np.ones([1,dim])
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测   
        fitness = CaculateFitness(X,fun) #计算适应度值
        
        bestII=np.argmin(fitness)
        Xbest = copy.copy(X[bestII,:])
        '''加入者位置更新'''
        for i in range(PDNumber+1,pop):
             if i>(pop - PDNumber)/2 + PDNumber:
                 X[i,:]= np.random.randn()*np.exp((Xworst - X[i,:])/i**2)
             else:
                 #产生-1，1的随机数
                 A = np.ones([dim,1])
                 for a in range(dim):
                     if(np.random.random()>0.5):
                         A[a]=-1       
                 AA = np.dot(A,np.linalg.inv(np.dot(A.T,A)))
                 X[i,:]= X[0,:] + np.abs(X[i,:] - GbestPositon)*AA.T
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测   
        fitness = CaculateFitness(X,fun) #计算适应度值
        '''意识到危险的麻雀更新'''
        Temp = range(pop)
        RandIndex = random.sample(Temp, pop) 
        SDchooseIndex = RandIndex[0:SDNumber]#随机选取对应比列的麻雀作为意识到危险的麻雀
        for i in range(SDNumber):
            if fitness[SDchooseIndex[i]]>BestF:
                X[SDchooseIndex[i],:] = Xbest + np.random.randn()*np.abs(X[SDchooseIndex[i],:] - Xbest)
            elif fitness[SDchooseIndex[i]] == BestF:
                K = 2*np.random.random() - 1
                X[SDchooseIndex[i],:] = X[SDchooseIndex[i],:] + K*(np.abs( X[SDchooseIndex[i],:] - X[-1,:])/(fitness[SDchooseIndex[i]] - fitness[-1] + 10E-8))
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测   
        fitness = CaculateFitness(X,fun) #计算适应度值
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve

'''F1函数'''
def F1(X):
    Results=np.sum(X**2)
    return Results
 
'''F2函数'''
def F2(X):
    Results=np.sum(np.abs(X))+np.prod(np.abs(X))
    return Results
 
'''F3函数'''
def F3(X):
    dim=X.shape[0]
    Results=0
    for i in range(dim):
        Results=Results+np.sum(X[0:i+1])**2
    return Results
 
'''F4函数'''
def F4(X):
    Results=np.max(np.abs(X))
    return Results
 
'''F5函数'''
def F5(X):
    dim=X.shape[0]
    Results=np.sum(100*(X[1:dim]-(X[0:dim-1]**2))**2+(X[0:dim-1]-1)**2)
    return Results
 
'''F6函数'''
def F6(X):
    Results=np.sum(np.abs(X+0.5)**2)
    return Results
 
'''F7函数'''
def F7(X):
    dim = X.shape[0]
    Temp = np.arange(1,dim+1,1)
    Results=np.sum(Temp*(X**4))+np.random.random()
    return Results
 
'''F8函数'''
def F8(X):
    Results=np.sum(-X*np.sin(np.sqrt(np.abs(X))))
    return Results
 
'''F9函数'''
def F9(X):
    dim=X.shape[0]
    Results=np.sum(X**2-10*np.cos(2*np.pi*X))+10*dim
    return Results
 
'''F10函数'''
def F10(X):
    dim=X.shape[0]
    Results=-20*np.exp(-0.2*np.sqrt(np.sum(X**2)/dim))-np.exp(np.sum(np.cos(2*np.pi*X))/dim)+20+np.exp(1)
    return Results
 
'''F11函数'''
def F11(X):
    dim=X.shape[0]
    Temp=np.arange(1,dim+1,+1)
    Results=np.sum(X**2)/4000-np.prod(np.cos(X/np.sqrt(Temp)))+1
    return Results
 
'''F12函数'''
def Ufun(x,a,k,m):
    Results=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<-a)
    return Results
 
def F12(X):
    dim=X.shape[0]
    Results=(np.pi/dim)*(10*((np.sin(np.pi*(1+(X[0]+1)/4)))**2)+\
             np.sum((((X[0:dim-1]+1)/4)**2)*(1+10*((np.sin(np.pi*(1+X[1:dim]+1)/4)))**2)+((X[dim-1]+1)/4)**2))+\
    np.sum(Ufun(X,10,100,4))
    return Results
 
'''F13函数'''
def Ufun(x,a,k,m):
    Results=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<-a)
    return Results
 
def F13(X):
    dim=X.shape[0]
    Results=0.1*((np.sin(3*np.pi*X[0]))**2+np.sum((X[0:dim-1]-1)**2*(1+(np.sin(3*np.pi*X[1:dim]))**2))+\
                 ((X[dim-1]-1)**2)*(1+(np.sin(2*np.pi*X[dim-1]))**2))+np.sum(Ufun(X,5,100,4))
    return Results
 
'''F14函数'''
def F14(X):
    aS=np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],\
                 [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
    bS=np.zeros(25)
    for i in range(25):
        bS[i]=np.sum((X-aS[:,i])**6)
    Temp=np.arange(1,26,1)
    Results=(1/500+np.sum(1/(Temp+bS)))**(-1)
    return Results
 
'''F15函数'''
def F15(X):
    aK=np.array([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
    bK=np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
    bK=1/bK
    Results=np.sum((aK-((X[0]*(bK**2+X[1]*bK))/(bK**2+X[2]*bK+X[3])))**2)
    return Results
 
'''F16函数'''
def F16(X):
    Results=4*(X[0]**2)-2.1*(X[0]**4)+(X[0]**6)/3+X[0]*X[1]-4*(X[1]**2)+4*(X[1]**4)
    return Results
 
'''F17函数'''
def F17(X):
    Results=(X[1]-(X[0]**2)*5.1/(4*(np.pi**2))+(5/np.pi)*X[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(X[0])+10
    return Results
 
'''F18函数'''
def F18(X):
    Results=(1+(X[0]+X[1]+1)**2*(19-14*X[0]+3*(X[0]**2)-14*X[1]+6*X[0]*X[1]+3*X[1]**2))*\
    (30+(2*X[0]-3*X[1])**2*(18-32*X[0]+12*(X[0]**2)+48*X[1]-36*X[0]*X[1]+27*(X[1]**2)))
    return Results
 
'''F19函数'''
def F19(X):
    aH=np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    cH=np.array([1,1.2,3,3.2])
    pH=np.array([[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],[0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
    Results=0
    for i in range(4):
        Results=Results-cH[i]*np.exp(-(np.sum(aH[i,:]*((X-pH[i,:]))**2)))
    return Results
 
'''F20函数'''
def F20(X):
    aH=np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    cH=np.array([1,1.2,3,3.2])
    pH=np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],\
                 [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    Results=0
    for i in range(4):
        Results=Results-cH[i]*np.exp(-(np.sum(aH[i,:]*((X-pH[i,:]))**2)))
    return Results
 
'''F21函数'''
def F21(X):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],\
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    Results=0
    for i in range(5):
        Results=Results-(np.dot((X-aSH[i,:]),(X-aSH[i,:]).T)+cSH[i])**(-1)
    return Results
 
'''F22函数'''
def F22(X):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],\
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    Results=0
    for i in range(7):
        Results=Results-(np.dot((X-aSH[i,:]),(X-aSH[i,:]).T)+cSH[i])**(-1)
    return Results
 
'''F23函数'''
def F23(X):
    aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],\
                  [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    Results=0
    for i in range(10):
        Results=Results-(np.dot((X-aSH[i,:]),(X-aSH[i,:]).T)+cSH[i])**(-1)
    return Results

OPT_algorithms = {'SSA':SSA}
OPT_algorithms.keys()

'''F7绘图函数'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
def F7(X):
    dim = X.shape[0]
    Temp = np.arange(1,dim+1,1)
    Results=np.sum(Temp*(X**4))+np.random.random()
 
    return Results
 
def F7Plot():
    fig = plt.figure(1) #定义figure
    ax = Axes3D(fig) #将figure变为3d
    x1=np.arange(-1.28,1.28,0.02) #定义x1，范围为[-1.28,1.28],间隔为0.02
    x2=np.arange(-1.28,1.28,0.02) #定义x2，范围为[-1.28,1.28],间隔为0.02
    X1,X2=np.meshgrid(x1,x2) #生成网格
    nSize = x1.shape[0]
    Z=np.zeros([nSize,nSize])
    for i in range(nSize):
        for j in range(nSize):
            X=[X1[i,j],X2[i,j]] #构造F7输入
            X=np.array(X) #将格式由list转换为array
            Z[i,j]=F7(X)  #计算F7的值
    #绘制3D曲面
    # rstride:行之间的跨度  cstride:列之间的跨度
    # rstride:行之间的跨度  cstride:列之间的跨度
    # cmap参数可以控制三维曲面的颜色组合
    ax.plot_surface(X1, X2, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
    ax.contour(X1, X2, Z, zdir='z', offset=0)#绘制等高线
    ax.set_xlabel('X1')#x轴说明
    ax.set_ylabel('X2')#y轴说明
    ax.set_zlabel('Z')#z轴说明
    ax.set_title('F7_space')
    plt.show()
 
F7Plot()