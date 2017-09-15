import argparse
import numpy
import sys
import logging
import time

logging.basicConfig(level=logging.ERROR)#,filename="logwithrandom.txt")

numpy.seterr(over="ignore")
#seed=int(sys.argv[5])
#seed=1014
#numpy.random.seed(seed)

class GAOptimizer(object):

    def __init__(self,feature_dim,pop=300,s=0.3,m=0.01):
        self.s_index=int(s*pop)
        self.m=m
        self.size=(pop,feature_dim)
        #print(self.s_index)
        self.population=numpy.random.random(size=(pop,feature_dim))/1.0
        self.best=None
        self.least_cost=9999999.0

    def gen_init(self):
        self.new_gen=numpy.zeros_like(self.population)

    def compute_output(self,X):
        return numpy.dot(X,self.population.T)

    def predictor_out(self,X):
        return numpy.dot(X,self.best.T)

    def mutator(self):
        random_mat=numpy.random.normal(size=self.size)
        mut_matrix_=numpy.random.binomial(1,self.m,size=self.size)
        mutators=numpy.multiply(mut_matrix_,random_mat)
        mask=(mutators==0)
        self.population=numpy.add(numpy.multiply(mask,self.population),mutators)

    def __reproduce(self,parents):
        indices=numpy.random.randint(self.s_index,size=2)
        _parents=parents.squeeze()[indices,:]
        _child=(_parents[0]+_parents[1])/2.0
        return _child

    def create_next_gen(self):
        #print(self.population.sum())
        self.gen_init()
        parents=self.population[:self.s_index]
        self.new_gen[:self.s_index]=parents
        for i in range(self.s_index,self.population.shape[0]):
            self.new_gen[i]=self.__reproduce(parents)
        self.population=self.new_gen
        self.mutator()

    def optimize(self,cost,val):
        logging.debug("cost :{} , val :{}".format(cost.mean(),val.mean()))
        indices=cost.argsort()
        self.population=self.population[indices]
        if val.mean() < self.least_cost:
            self.best=self.population[0]
        self.create_next_gen()

class LogisticRegression(object):

    def __init__(self,args):
        self.args=args

    def __sigmoid(self,X):
        return numpy.clip(1.0/(1.0+numpy.exp((-1)*(X))),0.000001,0.999999)

    def __cost(self,h,y):
        cost=(numpy.dot((-1)*y.T,numpy.log(h))-numpy.dot((1-y).T,numpy.log(1-h)))
        return cost

    def acc_cost(self,h,y):
        return numpy.dot(h.T,y)

    def __output(self,X):
        return self.optimizer.compute_output(X)

    def __gradient(self,h,y,X):
        return numpy.dot(X.T,(h-y))+((self.l/2)*self.W)

    def predict(self,X):
        output=self.__sigmoid(self.optimizer.predictor_out(X))
        return (output[:]>0.5)

    def accuracy(self,h,y):
#        print(h.shape)
#        print(y.shape)
        h=h[:,numpy.newaxis]
        total=y.shape[0]
        result=numpy.hstack((h,y))
        right=result[result[:,0]==result[:,1]].shape[0]
        return float(right)/float(total)

    def fit(self,X,y):
        feature_dim=X.shape[1]
        y=y[:,numpy.newaxis]
        train_X,train_y,test_X,test_y=self.create_train_test(X,y)
        self.optimizer=GAOptimizer(train_X.shape[1],self.args['population'],self.args['survival'])
        epoch=1
        min_cost=0.0
        prev_grad=0
        patience=5
        while epoch:
            output=self.__sigmoid(self.__output(train_X))
            cost=self.__cost(output,train_y)
            val_cost=self.__cost(self.__sigmoid(self.__output(test_X)),test_y)
            self.optimizer.optimize(cost.squeeze(),val_cost.squeeze())
            if epoch % 10 == 0:
                logging.info("Accuracy after {} epochs is {}".format(epoch,self.accuracy(self.predict(test_X),test_y)))
            if epoch == self.args['generations']:
                break
            epoch+=1

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
            pass
    return vector

def scale(X):
    return X/(1+numpy.max(X))

def tf(X):
    num_docs=numpy.apply_along_axis(lambda x:x[x!=0].shape[0],0,X)+1
    idf=numpy.log(X.shape[0]/num_docs)
    tf=numpy.apply_along_axis(lambda x:x/numpy.max(x),1,X)
    return tf*idf

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
    return tf(X) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 2",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment2.py [train-data] [train-label] [test-data] <optional args>")
    parser.add_argument("paths", nargs = 4)
    parser.add_argument("-n", "--population", default = 100, type = int,
        help = "Population size [DEFAULT: 100].")
    parser.add_argument("-s", "--survival", default = 0.3, type = float,
        help = "Per-generation survival rate [DEFAULT: 0.3].")
    parser.add_argument("-m", "--mutation", default = 0.01, type = float,
        help = "Point mutation rate [DEFAULT: 0.01].")
    parser.add_argument("-g", "--generations", default = 100, type = int,
        help = "Number of generations to run [DEFAULT: 100].")
    parser.add_argument("-r", "--random", default = -1, type = int,
        help = "Random seed for debugging [DEFAULT: -1].")
    args = vars(parser.parse_args())

    # Do we set a random seed?
    if args['random'] > -1:
        numpy.random.seed(1012)

    # Read in the training data.
    #X, y = _load_train(args["paths"][0], args["paths"][1])
    logging.info("Loading data.")
    X=get_dataset(args['paths'][0])
    enforce_shape=X.shape
    logging.info("Data loaded.")
    labels=load_data(args['paths'][1])
#    print(X.shape)
#    print(labels.shape)
    LR=LogisticRegression(args=args)
    LR.fit(X,labels)
    logging.info("Beginning test.")
    X_test=get_dataset(args['paths'][2],enforce_shape)
#    y_test=load_data(args['paths'][3])
#    print(X_test.shape)
#    print(y_test.shape)
#    logging.error("Testing set accuracy is {} for seed {}".format(LR.accuracy(LR.predict(X_test),y_test),seed))
    out=LR.predict(X_test)
    for i in out:
        print(int(i))
        pass
