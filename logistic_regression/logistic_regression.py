#!/usr/bin/env python3

import numpy
import sys

import logging

logging.basicConfig(level=logging.ERROR)

numpy.seterr(over="ignore")
#seed=int(sys.argv[5])
numpy.random.seed(1005)

class LogisticRegression(object):

    def __init__(self,alpha=0.001,l=0.001):
        self.alpha=alpha
        self.l=l
        self.W=None
        self.b=None

    def __sigmoid(self,X):
        return numpy.clip(1.0/(1.0+numpy.exp((-1)*(X))),0.000001,0.999999)

    def __cost(self,h,y):
        return numpy.mean(numpy.dot((-1)*y.T,numpy.log(h))-numpy.dot((1-y).T,numpy.log(1-h)))

    def __output(self,X):
        return numpy.dot(X,self.W)+self.b

    def __gradient(self,h,y,X):
        return numpy.dot(X.T,(h-y))+((self.l/2)*self.W)

    def predict(self,X):
        output=self.__sigmoid(self.__output(X))
        return (output[:]>0.5)

    def accuracy(self,h,y):
        total=y.shape[0]
        result=numpy.hstack((h,y))
        right=result[result[:,0]==result[:,1]].shape[0]
        return float(right)/float(total)

    def fit(self,X,y):
        feature_dim=X.shape[1]
        y=y[:,numpy.newaxis]
        train_X,train_y,test_X,test_y=self.create_train_test(X,y)
        # Glorot initialization.
        #_min=-4.0*numpy.sqrt(6.0/(feature_dim+1))
        _min=0
        _max=4.0*numpy.sqrt(6.0/(feature_dim+1))
        #self.W=numpy.random.uniform(size=(feature_dim,1),low=_min,high=_max)
        self.W=numpy.zeros(shape=(feature_dim,1))
        self.b=numpy.zeros(shape=(1,))
        epoch=1
        prev_cost=0.0
        while epoch:
            prev_self_W=self.W
            output=self.__sigmoid(self.__output(train_X))
            cost=self.__cost(output,train_y)
            val_cost=self.__cost(self.__sigmoid(self.__output(test_X)),test_y)
            #logging.debug("prev {} now {}".format(prev_cost,val_cost))
            if prev_cost < val_cost and epoch != 1:
                logging.info("Alpha is now {}".format(self.alpha))
                self.W=prev_self_W
                self.alpha*=0.1
                #pass
            if self.alpha < 0.0001 or epoch > 10000:
                #self.W=prev_self_W
                break
            grad=self.__gradient(output,train_y,train_X)
            self.b=self.b-(self.alpha*numpy.mean(output-train_y))
            self.W=self.W-(self.alpha*grad)
            if epoch % 100 == 0:
                accuracy=self.accuracy(self.predict(test_X),test_y)
                logging.info("Epoch {} error {}. Training error {}. Accuracy {}".format(epoch,val_cost,cost,accuracy))
            epoch+=1
            prev_cost=val_cost

    def create_train_test(self,X,y,split=0.1):
        dataset=numpy.hstack((X,y))
        numpy.random.shuffle(dataset)
        split_ind=int(split*X.shape[0])
        test_set=dataset[:split_ind]
        train_set=dataset[split_ind:]
        train_X,train_y=numpy.split(train_set,[-1],axis=1) 
        test_X,test_y=numpy.split(test_set,[-1],axis=1)
        return train_X,train_y,test_X,test_y


def load_data(filename):
    return numpy.loadtxt(filename,dtype=numpy.uint32,delimiter=" ")

def get_doc_data(data_set,doc_id):
    return data_set[data_set[:,0]==doc_id]

def create_feature_vector(word_counts,max_len):
    vector=numpy.zeros(max_len)
    for word_count in word_counts:
        try:
            vector[word_count[1]]=word_count[2]
        except:
            vector[-1]=word_count[2]
    #logging.debug("{}".format(vector))
    return vector

def scale(X):
    return X/(1+numpy.max(X))

def get_dataset(filename,enforce=None):
# load dataset
    logging.info("Loading data.")
    data=load_data(filename)
    logging.info("Data loaded.")
# get word count for each document
    doc_features=list()
    logging.info("Extracting features.")
    for i in numpy.unique(data[:,0]):
        doc_words=get_doc_data(data,i)
        #logging.debug("{}".format(doc_words))
        doc_features.append((i,doc_words))
    logging.info("Features extracted.")
    
# construct dataset matrix
    vectors=list()
    logging.info("Creating vectors.")
    if enforce == None:
        for _id,features in doc_features:
            vectors.append(create_feature_vector(features,numpy.max(data[:,1])+2)) # +1 to compensate for off by one and +1 to account for unseen words.
    else:
        for _id,features in doc_features:
            vectors.append(create_feature_vector(features,enforce[1])) 
    logging.info("Vectors created.")
    logging.info("Collecting to dataset matrix.")
    X=numpy.array(vectors)
    X=numpy.apply_along_axis(scale,0,X)
    return X 

if __name__=="__main__":

# load dataset
    logging.info("Loading data.")
    X=get_dataset(sys.argv[1])
    enforce_shape=X.shape
    logging.info("Data loaded.")
    labels=load_data(sys.argv[2])
  #print(X)
    #sys.exit()
# learn
    LR=LogisticRegression(alpha=0.01)
    LR.fit(X,labels)
    logging.info("Beginning test.")
    X_test=get_dataset(sys.argv[3],enforce_shape)
#    y_test=load_data(sys.argv[4])
#    logging.error("Testing set accuracy is {} for seed {}".format(LR.accuracy(LR.predict(X_test),y_test[:,numpy.newaxis]),seed))
    out=LR.predict(X_test)
    for i in out[:,0]:
        print(int(i))
        pass
