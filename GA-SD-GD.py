
# coding: utf-8

# In[3]:


import random
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
np.random.rand()


# In[4]:


def individual(popSize):
    return list(np.random.choice([0,1],popSize))


# In[5]:


def population (popSize,chromLeng):
    return [individual(popSize) for x in range(chromLeng)]

pop=population(4,5)
print(pop)


# In[8]:


def fitness(x1,x2):
   
    f=8-(x1+0.0317)**2+(x2)**2
    
    return f

def standDecoding(pop,precisions,minx,maxx):
    sumindv=0
    standardx1=0
    standardx2=0 
    sumindv1=0
    sumindv2=0
    
    fit=0
    standFit=[]
    for i in range(len(pop)):
        indv=pop[i]
        #standerd decoding
        for j in range(0,precisions-1):
            sumindv1=sumindv1+(indv[j]*2**(precisions-1-j))
        standardx1=minx+sumindv1/2**(precisions*(maxx-minx))
        for j in range(precisions,len(indv)):
            sumindv2+=(indv[j]*2**((len(indv)-precisions)-1-j))
        standardx2=minx+sumindv2/2**(precisions)*(maxx-minx)
        fit=fitness(standardx1,standardx2)
        
        if standardx1+standardx2==1:
            newfit=fit-abs(standardx1+standardx2-1)
            standFit.append(newfit)
        else:
            standFit.append(fit)
    return standFit

def GreyDecoding(pop,precisions,minx,maxx):
    summ1=0
    summ2=0
    sumind1=0
    sumind2=0
    greyx1=0
    greyx2=0
    grFit=[]
    for i in range(len(pop)):
        indv=pop[i]
        for j in range(0,precisions-1):
            for k in range(0,j):
                summ1+=indv[k]
            sumind1+=((summ1%2)*2**(precisions-1-j))
        greyx1=minx+sumind1/2**(precisions*(maxx-minx))
        
        for j in range(precisions,len(indv)):
            for k in range(0,j):
                summ2+=indv[k]
            sumind2+=((summ2%2)*2**((len(indv)-precisions)-1-j))
        greyx2=minx+sumind2/2**(precisions*(maxx-minx))
        calfit=fitness(greyx1,greyx2)
        
        if greyx1+greyx2==1:
            newfit=calfit-abs(greyx1+greyx2-1)
            grFit.append(newfit)
        else:
            grFit.append(calfit)
        
    return grFit



# In[9]:


stand=standDecoding(pop,3,-2,2)
print(stand)


# In[10]:


def evaRF (fitness):
    rfProb=[]
    sumOfFitness=0
    for i in range(len(fitness)):
        sumOfFitness+= fitness[i]
    for i in range(len(fitness)):
        rfProb.append(fitness[i]/sumOfFitness)
    return rfProb


# In[11]:


fit=evaRF(stand)
print(fit)


# In[12]:


def com(Rf):
    com_dist =[]
    count = 0
    for i in range(len(Rf)):
        if i==0:
            com_dist.append(Rf[i])
        else:
            count = Rf[i]+com_dist[i-1]
            com_dist.append(count)
    return com_dist


# In[13]:


comDist=com(fit)
print(comDist)


# In[14]:


def selection(comDist,Individuals):
    #print(randVar)
    selected=[]
    for i in range(len(comDist)):
        randVar=np.random.rand()
        for x in range(len(comDist)):
                if comDist[x] >= randVar:
                    selected.append(Individuals[x])
                    break
                else:
                    continue
                    
    return selected


# In[15]:


sel=selection(comDist,pop)
print(sel)


# In[16]:


def CrossOver(selectedIndv,popsize,probCrossOver=0.6):
    newPop=[]
    
    for i in  range(0,np.shape(selectedIndv)[0],2):
        if i>=popsize-1:
            break
        randVar= np.random.rand()
        s=i
        indv1=selectedIndv[s]
        indv2=selectedIndv[s+1]
        if randVar<probCrossOver:
            offSpr1=[]
            offSpr2=[]
            cutPiont=abs(np.round(np.random.rand()*popsize-1))
            for j in range(0,popsize):
                if j==cutPiont:
                    for x in range(j,popsize):
                        offSpr1.append(indv2[x])
                        offSpr2.append(indv1[x])
                    break
                else:
                    offSpr1.append(indv1[j])
                    offSpr2.append(indv2[j])
            
            newPop.append(offSpr1)
            newPop.append(offSpr2)
        else:
            
            newPop.append(indv1)
            newPop.append(indv2)

    return newPop



# In[17]:


newpop=CrossOver(sel,4,0.6)
print(newpop)


# In[18]:


def Mutation(newPop,popsize,probMut=0.05):
    mutPop=[]
    for i in range(len(newPop)):
        indv=newPop[i]
        for j in range(0,popsize):
            randVar= np.random.rand()
            if randVar<probMut:
                if indv[j]==0:
                    indv[j]=1
                else:
                    indv[j]=0
            else:
                continue
        mutPop.append(indv)
    return mutPop


# In[19]:


MutatedPopulation= Mutation(newpop,5,0.05)
print(MutatedPopulation)


# In[87]:


def elitism(fitness,pop):
    maxfit=0
    bestPop=[]
    for j in range(0,2,1):
        maxfit=max(fitness)
        for i in range(len(fitness)):
            if maxfit==fitness[i]:
                bestPop.append(pop[i]) 
                pop.remove(pop[i])
                fitness.remove(fitness[i])   
                break
            else:
                       continue
         
    return bestPop
    

def GA(popSize,numOfGeneration,chromLeng,precisions,minx,maxx,probCrossOver=0.6,probMut=0.05):
    FinalPop=[]
    best_hist=[]
    for i in range(0,numOfGeneration):
        Pop=population(chromLeng,popSize)
        #fitness=standDecoding(Pop,precisions,minx,maxx)
        fitness=GreyDecoding(Pop,precisions,minx,maxx)
        best=max(fitness)
        for i in range(len(fitness)):
            if best==fitness[i]:
                best_hist.append(fitness[i]) 
                break;
        
        
        el=elitism(fitness,Pop)
        
        RF = evaRF (fitness)
        
        comDist=com(RF)
        
        selectedIndv = selection(comDist,Pop)
        
        newPopulation= CrossOver(selectedIndv,chromLeng,probCrossOver)
    
        MutatedPopulation= Mutation(newPopulation,chromLeng,probMut)
        MutatedPopulation.append(el)
        FinalPop.append(MutatedPopulation)
    return FinalPop, best_hist


# In[88]:


final=GA(10,5,10,5,-2,2)
print(final[0])


# In[89]:


plt.plot(final[1])

