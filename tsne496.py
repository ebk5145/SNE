#tsne496.py

import numpy as np
import time
def P_calc(X=np.array([]),sigma=0.4):
    (n,d) = X.shape
    thisD=np.zeros([n,n])
    P=np.array([])
    for k in range(d):
        for i in range(n):
            for add in range(n):
                thisD[i,add] += np.abs(X[i,k] - X[add,k])**2 #produces a nxn matric with square norm of distance in each direction.
    P= np.exp(-thisD/2*(sigma**2))
    np.fill_diagonal(P,0)
    P= P/(np.sum(P,axis=1))
    np.fill_diagonal(P,1)
    return P.T

def P_bin(X=np.array([]),sigma=0.4):
    (n,d) = X.shape
    thisD=np.zeros([n,n])
    P=np.array([])
    for k in range(d):
        for i in range(n):
            for add in range(n):
                thisD[i,add] += np.abs(X[i,k] - X[add,k])**2 #produces a nxn matric with square norm of distance in each direction.
    P= np.exp(-thisD/2*(np.random.randn(100,1)**2))
    np.fill_diagonal(P,0)
    P= P/(np.sum(P,axis=1))
    np.fill_diagonal(P,1)
    return P.T

def bin_search(X,perplexity):
    """Uses binary search to find sigma for each x,y such that their H function is closest to the perplexity"""
    n,d=X.shape
    tol=.005
    sigma = np.array()
   

    while err>tol:
        top=100
        bottom=-100
        half=(top+bottom)/2
        opt=np.zeros([n,1])
        for i in range(n):
            sigma1[i] = np.arange(bottom,half)
            sigma2[i] = np.arange(half,top)
            P1=P_calc(X,sigma1)
            P2=P_calc(X,sigma2)
            H= -P*np.log(P)/np.log(2)
            err=np.abs(log(perplexity)-H)
            if P: #use this to remove half of the assigned sigma value by checking each row's Hdiff
                P+=P
    return P

def tsne(X=np.array([]),red_dim=2,sigma=0.4, max_iter =100):
    """
    Implements TSNE for given X and gives out a Y based on the following parameters:
    
    red_dim : The number of dimensions that data should be reduced to.
    sigma : The sigma that would be ideal to compute the probability matrix of Y.
    max_iter: The maximum iterations for the loop to run for finding the optimum neighbors.
    """
    t_init=time.time()
    #Defining Y based on the requested size of the reduced data.
    (n, d) = X.shape
    Y = np.random.randn(n,red_dim)
    P = P_calc(X,sigma)
    C_mat=[]
    for iter in range(max_iter):
        
        #Calculating Y and Q, initially based on the Random matrix
        Q = P_calc(Y,1/np.sqrt(2))
        Cf=P*np.log(P/Q)
        C=(Cf.sum(axis=1)).sum()
        C_mat.append(C)
        if iter%10 == 0:
            print('At iteration #',iter,' the Cost is: ', C)
            relv=C_mat[iter-1]-C_mat[iter] #Checks the relevance of the script by the amount of cost being decreased.
        
        # Computing the gradient
        A = np.array([Y[i,:] - Y for i in range(n)])
        probpart= P-Q+P.T-Q.T
        totgrad=A.T*probpart
        grad=2*totgrad.T.sum(axis=0)
        
        #Perform the update
        
        Y= Y + 0.5*grad + np.exp(-.15*iter)*np.random.random([n,red_dim])
       
        #Check need for the loop
        if C<5e-7:
            print('\n Script reached tolerance no more computation needed.')
            break
        elif relv<1e-8 and iter>10: #ends the loop if the relevance is less than the limit.
            print('\n This script reached its optimization limit')
            break
            
    Y= Y - np.mean(Y) #Centering the data.
    print('\n Computation time :', time.time()-t_init,'seconds')
    return Y         