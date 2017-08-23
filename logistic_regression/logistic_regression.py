#!/usr/bin/env python

import numpy
import sys

def load_train_data(filename):
    return numpy.loadtxt(filename,dtype=numpy.uint32,delimiter=" ")

def get_doc_data(data_set):
    return data_set[data_set[:,0]==0]

if __name__=="__main__":
    data=load_train_data(sys.argv[1])
    print(data[0,0])
    print(get_doc_data(data))
