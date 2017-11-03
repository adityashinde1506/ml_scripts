import numpy
import matplotlib.pyplot as plt

def generate_dataset(n=100): # This function generates the dataset.
    xi=numpy.random.uniform(-5,5,size=(1,n))
    ei=numpy.random.normal(0,0.1,size=(1,n))
    yi=numpy.sin(xi)+ei
    return xi,yi

def get_true_values(x): # This is the true function.
    return numpy.sin(x)

def loss(y,h_): # loss function
    return numpy.power(numpy.subtract(y,h_))

def gaussian(x,h):
    return (1/numpy.power((2*numpy.pi),0.5))*numpy.exp(-numpy.power(x,2)/h)

def kernel_smoother(x,xi,h,yi):
    _gaussian=gaussian(numpy.abs(x-xi)/h,h)
    return numpy.sum(yi*_gaussian)/numpy.sum(_gaussian)    

def loop_over_data(xi,yi,h):
    results=[]
    for data in xi[0]:
        results.append(kernel_smoother(data,xi,h,yi))
    return numpy.array(results)

if __name__=="__main__":
    x,y=generate_dataset(100)
    y_t=get_true_values(x)

# Sample training and testing set.
    train_x,train_y=generate_dataset(100)
    test_x,test_y=generate_dataset(100)
    h=[1.0,0.75,0.5,0.25,0.05,0.01,0.005,0.001]
    results={}
    for _h in h:
        results[_h]=loop_over_data(train_x,train_y,_h)
    loss={}
    for _h in results.keys():
        loss[_h]={}
        loss[_h]['empirical_error']=numpy.mean(loss(results[_h],train_y[0]))




