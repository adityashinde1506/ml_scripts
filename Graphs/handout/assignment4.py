import argparse
import numpy as np
import numpy

import sklearn.metrics.pairwise as pairwise

import sys

def read_data(filepath):
    Z = np.loadtxt(filepath)
    y = np.array(Z[:, 0], dtype = np.int)  # labels are in the first column
    X = np.array(Z[:, 1:], dtype = np.float)  # data is in all the others
    return [X, y]

def save_data(filepath, Y):
    np.savetxt(filepath, Y, fmt = "%d")

def iterate_rank(W,u,damp):
    prev_r=numpy.ones(u.shape)
    #print(numpy.linalg.norm(u))
    for i in range(100):
        new_r=(1-damp)*u+(damp*numpy.dot(W,prev_r))
        #new_r=new_r/numpy.linalg.norm(new_r)
        err=numpy.sum((new_r-prev_r)**2)
        #print("Err is {}".format(err))
        if err < epsilon:
            break
        prev_r=new_r
    return new_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 4",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment4.py -i <input-data> -o <output-file> [optional args]")

    # Required args.
    parser.add_argument("-i", "--infile", required = True,
        help = "Path to an input text file containing the data.")
    parser.add_argument("-o", "--outfile", required = True,
        help = "Path to the output file where the class predictions are written.")

    # Optional args.
    parser.add_argument("-d", "--damping", default = 0.95, type = float,
        help = "Damping factor in the MRW random walks. [DEFAULT: 0.95]")
    parser.add_argument("-k", "--seeds", default = 1, type = int,
        help = "Number of labeled seeds per class to use in initializing MRW. [DEFAULT: 1]")
    parser.add_argument("-t", "--type", choices = ["random", "degree"], default = "random",
        help = "Whether to choose labeled seeds randomly or by largest degree. [DEFAULT: random]")
    parser.add_argument("-g", "--gamma", default = 0.5, type = float,
        help = "Value of gamma for the RBF kernel in computing affinities. [DEFAULT: 0.5]")
    parser.add_argument("-e", "--epsilon", default = 0.01, type = float,
        help = "Threshold of convergence in the rank vector. [DEFAULT: 0.01]")

    args = vars(parser.parse_args())

    # Read in the variables needed.
    outfile = args['outfile']   # File where output (predictions) will be written. 
    _d = args['damping']         # Damping factor d in the MRW equation.
    k = args['seeds']           # Number of (labeled) seeds to use per class.
    t = args['type']            # Strategy for choosing seeds.
    gamma = args['gamma']       # Gamma parameter in the RBF kernel
    epsilon = args['epsilon']   # Convergence threshold in the MRW iteration.
    # For RBF, see: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel

    # Read in the data.
    X, y = read_data(args['infile'])
    A=pairwise.rbf_kernel(X,gamma=gamma)
    d=np.sum(A,axis=1)[:,numpy.newaxis]
    W=A/d
    classes=np.unique(y)
    R=[]
    for c in classes[classes >= 0]:
        if t=="random":
            u=np.zeros_like(y)
            indices=np.where(y==c)[0]
            for i in range(k):
                u[indices[np.random.randint(0,indices.shape[0])]]=1.
            u=u/numpy.linalg.norm(u)
            #print(u.shape)
        elif t=="degree":
            d=d.squeeze()
            u=numpy.zeros_like(y)
            indices=numpy.where(y==c)[0]
            degree=sorted(zip(indices,d[indices]),key=lambda x:x[1],reverse=True)
            best=[i[0] for i in degree][:k]
            #print(best)
            for index in best:
                u[index]=1
        #print(u)
        #print(A)
        rank=iterate_rank(W,u[:,numpy.newaxis],_d)
        #print("Shape of rank matrix is {}".format(rank.shape))
        #print(rank[0])
        R.append(rank)
    R=numpy.array(R).squeeze()
    #print(R[:,0])
    Out=numpy.apply_along_axis(numpy.argmax,0,R)[:,numpy.newaxis]
    save_data(outfile,Out)
