import numpy
import scipy
import sklearn.datasets
import matplotlib.pyplot as plt
import sys

def compute_Q(A,k):
    ohmega=numpy.random.standard_normal(size=(A.shape[1],k))
    Y=numpy.dot(A,ohmega)
    Q,R=scipy.linalg.qr(Y,mode='economic')
    return Q

def SSVD(A,Q):
    B=numpy.dot(Q.T,A)
    Uhat,sigma,Uhat_T=scipy.linalg.svd(B,full_matrices=False)
    return sigma # singular values
    
def compute_Q_oversample(A,k):
    return compute_Q(A,2*k)

if __name__=="__main__":
    data=sklearn.datasets.load_digits()
    data=data['data']
    for i in range(10):
        Q=compute_Q(data,20)
        Q_over=compute_Q_oversample(data,20)
        sigma_over=SSVD(data,Q_over)
        sigma=SSVD(data,Q)
        _,_sigma,_=scipy.linalg.svd(data,full_matrices=False)
        #_sigma=_sigma[:sigma.shape[0]]
        ssvd=(sigma,range(sigma.shape[0]))
        dsvd=(_sigma,range(_sigma.shape[0]))
        ossvd=(sigma_over,range(sigma_over.shape[0]))
        points=(ssvd,dsvd,ossvd)
        colors=('red','blue','green')
        groups=('SSVD','deterministic SVD','oversample SSVD')
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        for point,color,group in zip(points,colors,groups):
            x,y=point
            ax.scatter(y,x,c=color,alpha=0.5,linewidths=1)
        plt.title("Singular Values k=20")
        plt.legend(['SSVD','deterministic SVD','oversampled SSVD'],loc='upper right')
        #plt.show()
        plt.savefig("sing_values_"+str(i)+".png")
    
