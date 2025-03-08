nVar=50 #Número de Variables
import pandas as pd
import numpy as np
import random
from math import sin, cos, pi, exp, e, sqrt

matriz_fit = pd.read_csv('matriz_opt_pp.csv')
matriz_fit.columns = ['indice','g_proba','g_monto','g_ejecucion','leads','desembolso','marcaciones']

matriz_fit.indice = matriz_fit.indice.astype('int')
matriz_fit.g_ejecucion = matriz_fit.g_ejecucion.astype('int')
matriz_fit.desembolso = matriz_fit.desembolso.astype('int')
matriz_fit.marcaciones = matriz_fit.marcaciones.astype('int')

array_fit = matriz_fit[['indice','g_ejecucion','desembolso','marcaciones']].values
#Funciones objetivo: 
#F1 : Desembolso (ventas), según el grupo de probabilidad, monto de oferta y grupo de ejecución. (maximizar)
#F2 : Número de llamadas, según el grupo de ejecución y el número de leads en ese grupo. (minimizar)
# Toda la información ha sido recopilada en una matriz previamente, se da un pequeño castigo a la función si un grupo
# de ejecución se repite más de 12 veces

def fobj(individual):
    desemb = 0
    for k in range(len(individual)):
        desemb=desemb+int(array_fit[np.where((array_fit[:,0]==k+1) & (array_fit[:,1] ==individual[k]))][:,2])
    f1 = 1/(desemb/4000000)
    
    llamadas = 0
    for k in range(len(individual)):
        llamadas=llamadas+int(array_fit[np.where((array_fit[:,0]==k+1) & (array_fit[:,1] ==individual[k]))][:,3])
    f2 = llamadas/40000000
    
    if any(np.unique(individual, return_counts=True)[1]>12):
        f1=f1*1.5
        f2=f2*2
    
    return f1, f2

class EmptyParticle:

    def __init__(self):  
        self.Position=[]
        self.Velocity=[]
        self.Cost=[]
        self.Dominated=False
        self.Best_Position=[]
        self.Best_Cost=[]
        self.GridIndex=[]
        self.GridSubIndex=[]
        
def CreateEmptyParticle():   
    return EmptyParticle() 

def CreateWolfs(n):
    
    wolfs=[EmptyParticle() for i in range(n)]
    
    return wolfs

def Dominates(x,y):

    x=x.Cost
    y=y.Cost
    dom= (all(i<=j for i, j in zip(x, y)) & any(i<j for i, j in zip(x, y)))

    return dom

def DetermineDomination(pop):
    npop=len(pop)
    for i in range(len(pop)):
        pop[i].Dominated=False
        for j in range(0,i-1):
            if not pop[j].Dominated:
                if Dominates(pop[i],pop[j]):
                    pop[j].Dominated=True
                elif Dominates(pop[j],pop[i]):
                    pop[i].Dominated=True
                    break;
    return pop

def EliminarDuplicados(rep):
    total=[]
    rep2 = []
    for i in range(len(rep)):
        if list(rep[i].Position) not in [list(item) for item in total]:
            rep2.append(rep[i])
        total.append(rep[i].Position)
    return rep2

def GetNonDominatedParticles(pop):
    nd_pop=[]
    for i in range(len(pop)):
        if not pop[i].Dominated:
            nd_pop.append(pop[i])
    
    nd_pop = EliminarDuplicados(nd_pop)
    
    return nd_pop

def GetCosts(A):
    costs = []
    for i in range(len(A)):
        costs.append(A[i].Cost)
    
    return costs

def Cost_normalize(item,costs):
    
    if max(costs)!=min(costs):
        item_norm = (item- min(costs))/(max(costs)-min(costs))
    else:
        item_norm = (item- min(costs))/(np.mean(costs)) 
    
    return item_norm
    
    
def CreateHypercubes(costs,ngrid,alpha):

    nobj=2
    G = [[[],[]],[[],[]]]
    
    cost1= [item[0] for item in costs]
    cost1_norm = [Cost_normalize(item,cost1) for item in cost1]


    cost2= [item[1] for item in costs]  
    cost2_norm = [Cost_normalize(item,cost2) for item in cost2]

        
    costs = list(zip(cost1_norm,cost2_norm))
    
    for j in range(nobj):
        
        min_cj=min([i[j] for i in costs])
        max_cj=max([i[j] for i in costs])
        
        dcj=alpha*(max_cj-min_cj)
        
        min_cj=min_cj-dcj
        max_cj=max_cj+dcj
        
        gx=list(np.linspace(min_cj,max_cj,ngrid-1))
        
        G[j][0]=[-np.inf, *gx]
        G[j][1]=[*gx, np.inf]
        

    return G

def sub2ind(array_shape, rows, cols):
    return cols*array_shape[1] + rows+1

def GetGridIndex(particle,G):
    
    c = particle.Cost
    
    costs = GetCosts(GreyWolves)
    
    c = (Cost_normalize(c[0],[item[0] for item in costs]),Cost_normalize(c[1],[item[1] for item in costs]))
    
    nobj=2
    ngrid=len(G[0][0])
    SubIndex=[0,0]
    
    for j in range(nobj):
        U = G[j][1]
        i = next(x[0] for x in enumerate(U) if x[1] > c[j])
        SubIndex[j] = i
    
    Index = sub2ind([ngrid,ngrid],SubIndex[0],SubIndex[1])
    
    return Index,SubIndex

def GetOccupiedCells(pop):
    
    GridIndices = []
    for i in range(len(pop)):
        GridIndices.append(pop[i].GridIndex)
    
    occ_cell_index=list(set(GridIndices))
    
    occ_cell_member_count=list(np.zeros(shape=len(occ_cell_index)))

    for k in range(len(occ_cell_index)):
        occ_cell_member_count[k]=GridIndices.count(occ_cell_index[k])
    
    return occ_cell_index,occ_cell_member_count

def RouletteWheelSelection(p):

    r=np.random.random()
    c=np.cumsum(p)
    i=next(x[0] for x in enumerate(c) if x[1] >= r)

    return i

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def SelectLeader(rep,beta):
    
    if rep==[]:
        rep.append(GreyWolves[round(random.uniform(0, GreyWolves_num))-1])
        beta=1
    
    occ_cell_index,occ_cell_member_count = GetOccupiedCells(rep)
    p=[ x**(-beta) for x in occ_cell_member_count ]
    p=[ x/sum(p) for x in p ]
    
    selected_cell_index=occ_cell_index[RouletteWheelSelection(p)]
    
    GridIndices = []
    for i in range(len(rep)):
        GridIndices.append(rep[i].GridIndex)
    
    selected_cell_members=find(GridIndices, lambda x: x == selected_cell_index)
    n=len(selected_cell_members)
    
    selected_member_index=round(random.uniform(1, n))
    
    h=selected_cell_members[selected_member_index-1]
    
    rep_h=rep[h]
    
    return rep_h

def DeleteFromRep(rep,EXTRA,gamma):
    
    if rep==[]:
        rep.append(GreyWolves[round(random.uniform(0, GreyWolves_num))-1])
        gamma=1
    
    for k in range(EXTRA):
    
        occ_cell_index,occ_cell_member_count = GetOccupiedCells(rep)
        p=[ x**(-gamma) for x in occ_cell_member_count ]
        p=[ x/sum(p) for x in p ]

        selected_cell_index=occ_cell_index[RouletteWheelSelection(p)]

        GridIndices = []
        for i in range(len(rep)):
            GridIndices.append(rep[i].GridIndex)

        selected_cell_members=find(GridIndices, lambda x: x == selected_cell_index)
        n=len(selected_cell_members)

        selected_member_index=round(random.uniform(1, n))

        j=selected_cell_members[selected_member_index-1]

        rep.pop(j)
    
    return rep

lb = 1 # Límite inferior
lb = [lb] * nVar 

ub = 5 # Límite Superior
ub = [ub] * nVar

GreyWolves_num=200 # Número de Lobos 
MaxIt=75  # Numero de iteraciones
Archive_size=200   # Tamaño de soluciones de pareto

alpha=0.05  # Parametro de inflacion de Grid
nGrid=20  # Numero de Grids
beta=7    # Parametro de presión de selección de líderes
gamma=2    # Parametro de presión de selección de líderes extras

GreyWolves=CreateWolfs(GreyWolves_num)

for i in range(GreyWolves_num):
    GreyWolves[i].Velocity=0
    GreyWolves[i].Position=np.zeros(nVar)
    for j in range(nVar):
        GreyWolves[i].Position[j]=round(random.uniform(1,5))
    GreyWolves[i].Cost=fobj(GreyWolves[i].Position)
    GreyWolves[i].Best_Position=GreyWolves[i].Position
    GreyWolves[i].Best_Cost=GreyWolves[i].Cost

GreyWolves=DetermineDomination(GreyWolves)
Archive = GetNonDominatedParticles(GreyWolves)
Archive_costs=GetCosts(Archive)
G = CreateHypercubes(Archive_costs,nGrid,alpha)

for i in range(len(Archive)):
    Archive[i].GridIndex,Archive[i].GridSubIndex=GetGridIndex(Archive[i],G)

costs_graph = pd.DataFrame(columns=['x1','x2','n_iter'])

for it in range(MaxIt):
    
    a=2-it*((2)/MaxIt)
    for i in range(GreyWolves_num):
        Delta=SelectLeader(Archive,beta)
        Beta=SelectLeader(Archive,beta)
        Alpha=SelectLeader(Archive,beta)
        Archive = EliminarDuplicados(Archive)
        
        rep2=[]
        rep3=[]

        if len(Archive)>1:
            counter=0
            for newi in range(len(Archive)):
                if list(Delta.Position)!=list(Archive[newi].Position):
                    counter=counter+1
                    rep2.append(Archive[newi])
            Beta=SelectLeader(rep2,beta)


        if len(Archive)>2:
            counter=0
            for newi in range(len(rep2)):
                if list(Beta.Position)!=list(rep2[newi].Position):
                    counter=counter+1
                    rep3.append(rep2[newi])
            Alpha=SelectLeader(rep3,beta)
        
         
        c=2*np.random.rand(nVar)
        D=abs(c*Delta.Position-GreyWolves[i].Position)
        A=2*a*np.random.rand(nVar)-a
       
        X1=Delta.Position-A*abs(D)
        
        c=2*np.random.rand(nVar)    
        D=abs(c*Beta.Position-GreyWolves[i].Position)     
        A=2*a*np.random.random()-a
       
        X2=Beta.Position-A*abs(D)
        
        c=2*np.random.rand(nVar)  
        D=abs(c*Alpha.Position-GreyWolves[i].Position)
        A=2*a*np.random.random()-a
      
        X3=Alpha.Position-A*abs(D)
        
        GreyWolves[i].Position=np.around((X1+X2+X3)/3)
        
        GreyWolves[i].Position=[max(ub) if i >max(ub) else i for i in GreyWolves[i].Position]
        GreyWolves[i].Position=[min(lb) if i <min(lb) else i for i in GreyWolves[i].Position]
        
        GreyWolves[i].Cost=fobj(GreyWolves[i].Position)
        
    GreyWolves=DetermineDomination(GreyWolves)
    non_dominated_wolves=GetNonDominatedParticles(GreyWolves)
    Archive.extend(non_dominated_wolves)
    Archive=DetermineDomination(Archive)
    Archive=GetNonDominatedParticles(Archive)
    Archive = EliminarDuplicados(Archive)
    
    for i in range(len(Archive)):
        Archive[i].GridIndex,Archive[i].GridSubIndex=GetGridIndex(Archive[i],G)
    
    if len(Archive)>Archive_size:
        EXTRA=len(Archive)-Archive_size
        Archive=DeleteFromRep(Archive,EXTRA,gamma)
        
        Archive_costs=GetCosts(Archive)
        G=CreateHypercubes(Archive_costs,nGrid,alpha)
    
    costs=GetCosts(GreyWolves)
    Archive_costs=GetCosts(Archive)
    temp = pd.DataFrame(costs,columns=['x1','x2'])
    temp['n_iter'] = it
    costs_graph=costs_graph.append(temp)
    print('iteracion:',it, ' Cost:',min(Archive_costs))
    
total = []
for i in range(len(Archive)):
    total.append(np.array(Archive[i].Position))
len(np.unique(total,axis=0))

np.unique(total,axis=0)

for i in range(len(np.unique(total,axis=0))):
    solution = np.unique(total,axis=0)[i]

    desemb_final = 0
    for k in range(len(solution)):
        desemb_final=desemb_final+int(array_fit[np.where((array_fit[:,0]==k+1) & (array_fit[:,1] ==solution[k]))][:,2])

    llamadas_final = 0
    for k in range(len(solution)):
        llamadas_final=llamadas_final+int(array_fit[np.where((array_fit[:,0]==k+1) & (array_fit[:,1] ==solution[k]))][:,3])

    print('Solución nº',i+1,'/ Desembolso:',desemb_final/4,' Llamadas:',llamadas_final/4)
    
np.unique(solution, return_counts=True)[1]

import plotly_express as px

graph_x_min = 0.04
graph_x_max = 0.12
graph_y_min = 0.03
graph_y_max = 0.09

px.scatter(
    costs_graph,
    x       = "x1",
    y       = "x2",
    range_x = [graph_x_min, graph_x_max],
    range_y = [graph_y_min, graph_y_max],
    animation_frame = "n_iter"
)
