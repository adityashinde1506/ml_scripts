import numpy
import sys

L=numpy.loadtxt(sys.argv[1])
T=numpy.loadtxt(sys.argv[2])
right=len(L[L==T])
print(right)
print(len(T))
