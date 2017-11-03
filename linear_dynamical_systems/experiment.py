import argparse
import numpy
from scipy.linalg import pinv, svd # Your only additional allowed imports!
import matplotlib.pyplot as plotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 3",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment3.py <arguments>")
    parser.add_argument("-f", "--infile", required = True,
        help = "Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required = True, type = int,
        help = "Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required = True,
        help = "Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # Collect the arguments.
    input_file = args['infile']
    q = args['dimensions']
    output_file = args['output']

    # Read in the dynamic texture data.
    M = numpy.load(input_file)
    #plotter.imshow(M[-1])
    #plotter.show()
    imshape=(M.shape[1],M.shape[2])
    M=M.reshape((M.shape[0],-1)).T
    U,S,V=svd(M,full_matrices=False)
    #print(U.shape)
    C=U[:,:q]
    S=numpy.diag(S[:q])
    V=V[:q]
    X=numpy.dot(S,V)
    print(X.shape)
    print(C.shape)
    #X=M.T
    X1=X[:,:-1]
    X2=X[:,1:]
    X2=pinv(X2)
    A=numpy.dot(X1,X2)
    last_step=X[:,-1]
    next_step=numpy.dot(A,last_step)
    image=numpy.dot(C,next_step)
    image=image.reshape(imshape)
    #plotter.imshow(image)
    #plotter.show()
    #C=numpy.dot(M,pinv(numpy.eye(176)))
    #numpy.save(output_file,image)
