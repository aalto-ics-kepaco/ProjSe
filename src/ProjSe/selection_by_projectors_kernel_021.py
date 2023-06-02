######################
## Version 0.21 #######
## /**********************************************************************
##   Copyright 2023, Sandor Szedmak  
##   email: sandor.szedmak@uibk.ac.at
##          szedmak777@gmail.com
##
##    This file contains the code for Variable selection by projection
##    operators(VASP).
##
##     MIT License
##     Copyright (c) 2023 KEPACO
##
##     Permission is hereby granted, free of charge, to any person obtaining
##     a copy of this software and associated documentation files (the
##     "Software"), to deal in the Software without restriction, including
##     without limitation the rights to use, copy, modify, merge, publish,
##     distribute, sublicense, and/or sell copies of the Software, and to
##     permit persons to whom the Software is furnished to do so, subject
##     to the following conditions:
##
##     The above copyright notice and this permission notice shall be
##     included in all copies or substantial portions of the Software.
##
##     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##     TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
## 
## ***********************************************************************/
######################
import sys

import numpy as np
## import pylab as plt
## ##########################################
## from mmr import mmr_normalization_new
## #################################
class cls_projector_kern:
  """
  $\phi: \amthcal{X}\rightarrow \H_{\phi}$ feature mapping
  if $\phi(X)$ is the row wise feature representation of a set of vectors
  $x_1,\dots,x_k$ spanning subsapce $L_X$ then
  the projector into $L_X$ is given by
  $$P_{L_X}$=\phi(X)'(\phi(X)\phi(X)')^{-1}\phi(X)
            =\phi(X)'K_{\phi}^{-1}\phi(X)
  $$
  """

  ## --------------------------  
  def __init__(self, func_kernel = None, **kernel_params):
    """
    Input:  func_kern    reference to kernel function, 
                         it assumes two 2d array X1,X2 inputs, a 2d array of kernel matrix
                         The inner product are computed between the rows of X1 and X2.
                         The number of rows in X1 and X2 can be different.
                         See the default example: self.lin_kern in this class 
                         If func_kern == None then linear kernel is used.
     """

    self.Y=None         ## reference frame
    self.X=None         ## target frame
    self.xmask=None     ## indicator of selected x vectors up to t

    self.Ky=None        ## kernel of references
    self.Kyx = None     ## output-input kernel
    self.Kx = None      ## input kernel, only computed if centralization is required
    self.DVT = None     ## the transpose of generalized inverse 
                        ## of the square root of Ky kernel

    if func_kernel is None:
      self.func_kern = self.lin_kern  ## linear kernel as default
    else:
      self.func_kern = func_kernel   ## reference to the kernel function 

    self.t = 0          ## iteration counter
    self.xmask = None   ## indicators of the selected variables         
    self.xstat = None  ## the scores of the selected variables

    self.kernel_params = kernel_params
    
    return

  ## ---------------------------
  def lin_kern(self,X1,X2, **kernel_params):
    """
    Task: to compute default linear kernel between rows, 
          for variables the inputs need to be transposed
    Input:  X1   2d array of left data
            X2   2d array of right data 
    Output: KX   2d array of kernel
    """

    tshape1 = X1.shape
    tshape2 = X2.shape

    if len(tshape1) == 1:
      X1 = X1.reshape((1,X1.shape[0]))
    if len(tshape2) == 1:
      X2 = X2.reshape((1,X2.shape[0]))

    KX = np.dot(X1,X2.T)

    return(KX)
  
  ## ---------------------------
  def norm_in_kern(self,X):
    """
    Task: to compute the norms of feature vectors  in an uncentralized kernel
    Input: X         2d array of data, exmaples in the rows
    Output: kxnorms  vector of norms
    """

    m,n = X.shape
    kxnorms=np.zeros(m)
    for i in range(m):
      kxnorms[i]=np.sqrt(self.func_kern(X[i:i+1],X[i:i+1], \
                         **self.kernel_params))
      if kxnorms[i]==0:
        kxnorms[i]=1

    return(kxnorms)

  ## -----------------------------
  def centr_kern(self,K):
    """
    Task:  to centralize kernels
    Input:   K    2d array of kernel
    Output:  K    centralized kernel
    """

    (m1,m2)=K.shape

    K=K-np.outer(np.ones(m1),np.mean(K,axis=0)) \
       -np.outer(np.mean(K,axis=1),np.ones(m2)) \
       +np.ones((m1,m2))*np.mean(K)

    return(K)

  ## --------------------------
  def load(self,Y,X, ilocal = 0, iscale = 1):
    """
    Task: to preprocess the kernels for variable selections
          load reference frame Y and target X
    Input: Y       2d array of references, output variables
           X       2d array of variables to select, input variables
           ilocal  =1 kernels are centralized, =0 not
           iscale  =1 kernels are normalized, =0 not
    Output: 
    Modifies: self.Ky, self.Kyx, self.DVT, self.t, self.xmask,self.xstat
    """

    (m,ny)=Y.shape
    nx=X.shape[1]

    func_kern = self.func_kern

    ## computer kernels between variables
    self.Ky=func_kern(Y.T,Y.T, **self.kernel_params)    ## output kernel
    self.Kyx=func_kern(Y.T,X.T, **self.kernel_params)   ## kernel between output and input 
    if ilocal == 1:                  ## centralization
      self.Ky=self.centr_kern(self.Ky)
      self.Kyx=self.centr_kern(self.Kyx)
      self.Kx=func_kern(X.T,X.T, **self.kernel_params)
      self.Kx=self.centr_kern(self.Kx)
      if iscale == 1:    ## normalization by L2 norm in the feature space
        kynorms = np.sqrt(np.diag(self.Ky))
        kxnorms = np.sqrt(np.diag(self.Kx))
        kynorms += (kynorms == 0)
        kxnorms += (kxnorms == 0)
        self.Ky /= np.outer(kynorms,kynorms)
        self.Kyx /= np.outer(kynorms,kxnorms)
    else:
      if iscale == 1:   ## normalization by L2 norm in the feature space
        kynorms = np.sqrt(np.diag(self.Ky))
        kynorms += (kynorms == 0)
        self.Ky /= np.outer(kynorms,kynorms)
        kxnorms = self.norm_in_kern(X.T)
        kxnorms += (kxnorms == 0)
        self.Kyx /= np.outer(kynorms,kxnorms)

    ## compute the right singular vectors and the singular values of the output data
    d_Y,V_Y = np.linalg.eigh(self.Ky)  
    ## to avoid division by 0
    eps = 10**(-6)   ## lower bound on eigen values
    ## to avoid division by zeros
    inz = np.where(d_Y>=eps)[0]   
    d_Y[inz]=1/d_Y[inz]
    inz = np.where(d_Y<eps)[0]   
    d_Y[inz]=0
    d_Y = np.sqrt(d_Y)

    ## the matrix factor containing transpose of the generalized inverse 
    ## of the square root of the kernel Ky 
    self.DVT = np.dot(np.diag(d_Y),V_Y.T)
    
    self.t=0                  ## iteration counter
    self.xmask=np.zeros(nx)   ## mask of the selected variables

    return

  ## -----------------------------------
  def full_cycle(self,Y,X,nitem,ilocal = 1, iscale = 1):
    """
    Task: to enumerate the x variables best correalting with Y
          but conditionaly uncorrelating with the previous selection
    Input:  Y   2d array reference set, variables in the columns
            X   2d array of variables to be selected variables in the columns
            nitem   maximum number of x variables selected
            ilocal  =1 X,Y cenralized =0 not
            iscale  =1 X,Y normalized row wise to have length 1 = 0 not
    Output  lorder  list of x variables aranged by selection order
    """

    self.load(Y,X,ilocal = ilocal, iscale = iscale)
    m,nx=X.shape
    ny=Y.shape[1]

    if nitem>=nx:
      nitem0=nx
    else:
      nitem0=nitem

    ## U = D_{Y}^{-1}V_{Y}^{T}\phi(Y)^{T}  
    ## P_{t}X^{T} = \tilde{U}\tilde{U}^{T}\phi(X)  
    ## half projection  = \tilde{U}^{T}\phi(X)
    ## ||P_{t}X^{T}||_2^2 = \phi(X)^{T}\tilde{U}\tilde{U}^{T}\phi(X)

    lorder = []

    ## \tilde{U}_0 = U_{Y} = \phi{Y}\mbf{V}_{Y}\mbf{D}_{Y}^{-1} 

    ## t = 0 case, no qt
    UtX = np.dot(self.DVT,self.Kyx)  ## half projections
    ## print('UtX.shape:',UtX.shape)

    self.xstat=np.zeros(nitem)   ## the score of the selected variables

    for t in range(nitem):
      ## print(t)
      ix = np.where(self.xmask==0)[0]       ## remaining input variables
      if t > 1:
        ## compute \tilde{U}_t^{T}\phi(X)^{T}  =  Q_t\tilde{U}_{t-1}^{T}\phi(X)^{T}
        ## Q_t = I - (q_t q_t^{T})||q_t||^2
        UtX = UtX - np.outer(qt,np.dot(qt,UtX))   ## projection into the complement
      ## print(np.max(np.abs(UtX)))
      n2UtX = np.sum(UtX**2,0)                  ## square norms
      ## print('max norm:',np.max(n2UtX))
      kstar = np.argmax(n2UtX)               ## find the most fitting
      ## kstar relative to the indexes in ix
      self.xstat[t] = n2UtX[kstar]           ## score
      self.xmask[ix[kstar]] = 1        ## include the selected variable 
                                       ## into the list of selected variables
      lorder.append(ix[kstar])         ## list of selected variables
      qt = UtX[:,kstar]           ## pick up the next qt
      ## @@@@@@@@@@@@@@@@@@@@@@@@@
      qnorm = np.sqrt(np.sum(qt**2))
      qnorm = qnorm +(qnorm==0)
      qt /= qnorm 
      ## @@@@@@@@@@@@@@@@@@@@@@@@@
      ## drop kstar column from UtX
      UtX = np.delete(UtX,np.s_[kstar:kstar+1],1) 

    return(lorder)
    
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
  Input:  X1     2d array 
          X2     2d array
          sigma  the standard deviation of the Gaussian
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

  ## test environment
  n=20                    ## number of variables
  m=1000                   ## number of examples  
  ny = 10                   ## output dimension
  rng = np.random.default_rng()

  X=rng.standard_normal(size=(m,n))  ## the variables to be selected in the columns
  X=centr_data(X)                      
  X=norm_data(X)
  W=rng.standard_normal(size=(n,ny))  ## linear mapping
  Y=np.dot(X,W)           ## reference variables 
  ## Y=centr_data(Y)
  ## Y=norm_data(Y)

  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ## the input variable selection procedure
  nitem=20                  ## number of variables to be selected
  kernel_params = { }     ## e.g.: '{'sigma': 1} 
  kernel_func = kern_func_gaussian
  ## construct the object
  cproject=cls_projector_kern(func_kernel = kernel_func, **kernel_params) 
  lorder=cproject.full_cycle(Y,X,nitem)  ## run the selection
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  print(lorder)

  ## selected variables
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
    print(ycorr)
    xstat_projection[i] = ycorr

    
  ## random selection
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
    print(ycorr)
    xstat_random[i] = ycorr

  ## projective prediction is red, random is blue
  ## plt.plot(xstat_random,'b',xstat_projection,'r')
  ## plt.show()

  print('Bye')
  
  return
  
## ################################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)
