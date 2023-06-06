######################
import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

## ##########################################
import ProjSe 

## ##########################################
## Utility functions
## ##########################################
def centr_data(X):
  """
  Task: data centralization by column mean
  """
  
  xmean = np.mean(X,0)
  X -= np.outer(np.ones(X.shape[0]),xmean)

  return(X)
## #################################
def norm_data(X):
  """
  Task: data normalization by L_2 norm row wise
  """
  
  xnorm = np.sqrt(np.sum(X**2,1))
  xnorm += 1*(xnorm==0)
  X /= np.outer(xnorm,np.ones(X.shape[1]))

  return(X)
## #################################
def kern_func_gaussian(X1,X2, **kernel_params):
  """
  Task:  to compute the Guassian kernel between the rows of the input arrays 
  Input:  X1               2d array 
          X2               2d array
          kernel_params    dictionary, e.g. { 'sigma' : 1 }
                           the standard deviation of the Gaussian
                           if it is None or 'sigma' is not in kernel_params
                           then sigma = np.sqrt(X1.shape[1]) is the default
  Output: KX12   Gaussian kernel        
  """

  m1=X1
  m2=X2.shape[0]

  if kernel_params is None:
    sigma = np.sqrt(X1.shape[1])
  else:
    if 'sigma' in kernel_params:
      sigma = kernel_params['sigma']
    else:
      sigma = np.sqrt(X1.shape[1])    
  
  ## compute norms
  if len(X1.shape)>1:
    m1=X1.shape[0]
    d1=np.sum(X1**2,axis=1)
  else:
    m1=1
    d1=np.sum(X1**2)
  if len(X2.shape)>1:
    m2=X2.shape[0]
    d2=np.sum(X2**2,axis=1)
  else:
    m2=1
    d2=np.sum(X2**2)
    
  e1=np.ones(m1)
  e2=np.ones(m2)

  K=np.dot(X1,X2.T)
  K=np.outer(d1,e2)+np.outer(e1,d2)-2*K
  KX12=np.exp(-K/(2*sigma**2))
  
  return(KX12)

## #################################
## #################################
def main(workmode):
  """
  Task: an example code of the projection based selection
  """

  igraph  = 1 ## =1 graph is created =0 not
  
  ## test environment
  n=100                    ## number of variables
  m=100000                   ## number of examples  
  ny = 20                   ## output dimension
  nitem=20                  ## number of variables to be selected

  rng = np.random.default_rng()

  ## data generation
  ## the variables to be selected in the columns
  X=rng.standard_normal(size=(m,n))  
  X=centr_data(X)                      
  X=norm_data(X)
  W=rng.standard_normal(size=(n,ny))  ## linear mapping
  Y=np.dot(X,W)           ## reference variables 
  ## Y=centr_data(Y)
  ## Y=norm_data(Y)

  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ## the input variable selection procedure
  ## linear kernel: {}, Gaussian:   '{'sigma': 1}
  kernel_params = { }
  ## linear kernel = None 
  kernel_func = kern_func_gaussian
  ## construct the object
  cproject=ProjSe.cls_projector_kern(func_kernel = kernel_func, \
    **kernel_params) 
  lorder=cproject.full_cycle(Y,X,nitem)  ## run the selection
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  print(lorder)

  ## test the selected variables
  xstat_projection = np.zeros(nitem)
  xstat_random = np.zeros(nitem)
  for i in range(nitem):
    Xl = X[:,lorder[0:i+1]]
    ## recompute the output prediction on the selected variables
    Q = np.linalg.pinv(np.dot(Xl.T,Xl))
    XTY = np.dot(Xl.T,Y)
    Wl = np.dot(Q,XTY)
    Yl = np.dot(Xl,Wl)
    ## accuracy measured by P-correlation
    ycorr = np.corrcoef(Y.ravel(),Yl.ravel())[0,1]
    print('P-corr. by Projector:',str('%7.4f'%ycorr))
    xstat_projection[i] = ycorr

    
  ## random selection for comparison
  xrandom = np.arange(nitem)
  rng.shuffle(xrandom)
  for i in range(nitem):
    Xl = X[:,xrandom[0:i+1]]
    ## recompute the output prediction on the selected variables
    Q = np.linalg.pinv(np.dot(Xl.T,Xl))
    XTY = np.dot(Xl.T,Y)
    Wl = np.dot(Q,XTY)
    Yl = np.dot(Xl,Wl)
    ## accuracy measured by P-correlation
    ycorr = np.corrcoef(Y.ravel(),Yl.ravel())[0,1]
    print('P-corr. by Random:',str('%7.4f'%ycorr))
    xstat_random[i] = ycorr


  if igraph == 1:

    fig = plt.figure(figsize=(5,5))
    fig.suptitle('Comparing projective selection to random')

    ax=plt.subplot2grid((1,1),(0,0),colspan=1,rowspan=1)
    ax.plot(xstat_random,label = 'Random')
    ax.plot(xstat_projection, label = 'Projector')
    ax.set_ylabel('Pearson correalation')
    ax.set_xlabel('Number of selected variables')
    ax.set_xticks([i for i in range(nitem)])
    ax.set_xticklabels([str(i+1) for i in range(nitem)], fontsize=10 )
    ax.set_title('Sample size:'+str('%6d'%m)+', '+'Variables:'+str('%4d'%n))
    
    ax.legend()
    ax.grid(True,ls='--')

    plt.tight_layout(pad=1)
    plt.show()

  print('Bye')
  
  return
  
## ################################################################
## in Jupyter this part of the code needs to be commented out
## ################################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)
