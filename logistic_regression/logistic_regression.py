#!/usr/bin/env python

import numpy
import sys

import logging

logging.basicConfig(level=logging.DEBUG)

class LogisticRegression(object):

    def __init__(self,alpha=0.001):
        self.alpha=alpha
        self.W=None
        self.b=None

    def __softmax(self,X):
        max_X=numpy.max(X)
        exp_X=numpy.exp(X-max_X) # To avoid numpy overflow.
        return numpy.divide(exp_X,numpy.sum(exp_X))

    def __sigmoid(self,X):
        return numpy.divide(1,1+numpy.exp(-X))

    def __cost(self,h,y):
        return 

    def predict(self,X):
        return numpy.apply_along_axis(self.__sigmoid,1,(numpy.add(numpy.dot(X,self.W),self.b)))

    def fit(self,X,y):
        feature_dim=X.shape[1]
        # Glorot initialization.
        _min=0
        _max=4.0*numpy.sqrt(6.0/(feature_dim+1))
        self.W=numpy.random.uniform(size=(feature_dim,1),low=_min,high=_max)
        self.b=numpy.random.uniform(size=(1,))
        output=self.predict(X)
        print(output)

def load_data(filename):
    return numpy.loadtxt(filename,dtype=numpy.uint32,delimiter=" ")

def get_doc_data(data_set,doc_id):
    return data_set[data_set[:,0]==doc_id]

def create_feature_vector(word_counts,max_len):
    vector=numpy.zeros(max_len)
    for word_count in word_counts:
        vector[word_count[1]]=word_count[2]
    return vector

def scale(X):
    return numpy.divide(numpy.subtract(X,numpy.mean(X)),numpy.var(X))

if __name__=="__main__":

# load dataset
    logging.info("Loading data.")
    data=load_data(sys.argv[1])
    logging.info("Data loaded.")
    labels=load_data(sys.argv[2])
# get word count for each document
    doc_features=list()
    logging.info("Extracting features.")
    for i in numpy.unique(data[:,0]):
        doc_words=get_doc_data(data,i)
        doc_features.append((i,doc_words))
    logging.info("Features extracted.")
    
# construct dataset matrix
    vectors=list()
    logging.info("Creating vectors.")
    for _id,features in doc_features:
        vectors.append(create_feature_vector(features,numpy.max(data[:,1])+1))
    logging.info("Vectors created.")
    logging.info("Collecting to dataset matrix.")
    X=numpy.array(vectors)
    X=numpy.apply_along_axis(scale,0,X)
    LR=LogisticRegression()
    LR.fit(X,labels)
