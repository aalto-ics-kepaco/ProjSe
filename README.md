# ProjSe-VaSP: Variable selection by projection operators

## Requirements

The application of the ProjSe assumes the  **Python** interpreter, version at least **3.7**, and the **numpy** package, version at least **1.20**. 

To run the examples also requires the **matplotlib**. All these packages
can be freely downloaded and installed from *pypi.org*. 

## Installation

The ProjSe package might be installed by the following procedures. 

### Directly from the github

>pip3 install git+https://github.com/aalto-ics-kepaco/ProjSe.git#egg=ProjSe

### Downloading from github

>mkdir projective_selection

>cd projective_selection

>git clone https://github.com/aalto-ics-kepaco/ProjSe

After downloading the ProjSe package it can be installed by the following command: 

>pip3 install projective_selection/ProjSe

Before installing the  ProjSe package the latest version of the Python packages **pip** and **build** need to be installed. 

>pip3 install --upgrade pip

>pip3 install --upgrade build


The ProjSe can be imported as

>import  ProjSe

## Running the projective selection algorithm

There is a demonstration in the "examples" directory. It requires the installation of the matplotlib.

## Interface:

The basic class definition:

cprojector = cls_projector_kern(func_kernel = None, **kernel_params)

> class cls_projector_kern:

> def __init__(self, func_kernel = None, **kernel_params):

>    """

>    Input:
  
>     func_kern
> reference to a kernel function, it assumes two 2d array X1,X2 inputs, 
a 2d array of kernel matrix.
>  The inner product are computed between the rows of X1 and X2. 
> The number of rows in X1 and X2 can be different. 
> See the default example: self.lin_kern in this class.  
If func_kern == None then the linear kernel is used.

>  kernel_params 

>  dictionary of parameters transferred to the function given in the func_kern.

>     """

Running the variable selection:

> cprojector.full_cycle(Y, X, nitem, ilocal = 1, iscale = 1) 
> 
>   def full_cycle(self,Y, X, nitem, ilocal = 1, iscale = 1):

>    """

>    Task: to enumerate the x variables best correlating with Y
>          but conditionally uncorrelating with the previous selection

>    Input:  

>            Y   2d array reference set, variables in the columns

>            X   2d array of variables to be selected variables in the columns

>            nitem   maximum number of x variables selected

>            ilocal  =1 X,Y centralized =0 not

>            iscale  =1 X,Y normalized row wise to have length 1 = 0 not

>    Output  lorder  list of x variables arranged by selection order

>    """

The selection score can be read out of cprojector.xstat which contains the scores in the order of selection.  






