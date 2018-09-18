#!/usr/bin/python3
import  csv
from math import sqrt 
import matplotlib.pyplot as plt
from random import seed
from random import randrange
from csv import reader
import numpy  as np

# vocab size
vocabulary_size = 8000


'''
Here is sample linear regression 
y=b0+b1*x

here b0 & b1 are coefficients , we must calculate from Training data

# formula for calculation 

b1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2 )
b0 = mean(y) - B1 * mean(x)

where the i refers to the value of the  value of the input x or output y.


#  calculate mean and variance 
# mean of x values can be calculated from 
# mean(x) = sum(x)/count(x)
'''
def  mean(values):
	return  sum(values)/int(len(values))

# calculating  variance 
#variance=sum( (x - mean(x))^2 )

def  variance(values,mean):
	return sum([(x-mean)**2  for x in values])

#  calculating  mean and variance

dataset=[]
with open('trafficnew.csv') as f:
	readf=csv.reader(f,delimiter=',')
	for row in  readf:
		if 'X' in row:
			continue
		else:
			value=[int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5])]
			dataset.append(value)
print(dataset)
z=dataset
x=[ir[0] for ir in  z]
y=[irr[1] for irr in  z]
x1=[irrr[2] for irrr in  z]
x2=[irrrr[3] for irrrr in  z]
print(x)
print("_____________________")
print(y)
# mean x and mean y
mean_x,mean_y=mean(x),mean(y)
var_x,var_y=variance(x,mean_x),variance(y,mean_y)

# printing information 
#print('x stats: mean=%.3f and variance=%.3f'%(mean_x,var_x))
#print('y stats: mean=%.3f and variance=%.3f'%(mean_y,var_y))

# building own RNN
class RNNNumpy:
     
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)) 




# forward propogation 

def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [s,o]

# calling  constructor  
RNNNumpy.forward_propagation = forward_propagation


#  prediction process

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)
 
RNNNumpy.predict = predict


#  creating  covar with loss function 
# covariance between x and y 
def  covariance(x,mean_x,y,mean_y):
        covar=0.0
        for i in range(len(x)):
                covar += (x[i] - mean_x)* (y[i] - mean_y)
        return covar

# calculate covariance 
cov=covariance(x,mean_x,y,mean_y)
print('covariance: %3f'%(cov))
#  Estimate coefficients
#b1=sum((x[i]-mean(x))*(y(i) - mean(y)))/sum((x(i) - mean(x))^2)
#b1=covariance(x,y)/variance(x)

#b0 = mean(y) - B1 * mean(x)
def coefficient(dataset):
        b1=cov/variance(x,mean_x)
        b0=mean_y-b1*mean_x
        return [b0,b1]
b0,b1=coefficient(dataset)
print('coefficients RNN  : b0=%.3f,b1=%.3f'%(b0,b1))


# making prediction 
#=b0+b1*x

#calculate root mean squared error 
def rmse_metric(actual,predicted):
        sum_error=0.0
        for i in range(len(actual)):
                prediction_error=predicted[i]-actual[i]
                sum_error += (prediction_error**2)
        mean_error=sum_error/float(len(actual))
        return sqrt(mean_error)

# evaluate regressoin algorithm on training datasets
def evaluate_algorithm(dataset,algorithm):
        test_set=list()
        for row in dataset:
                row_copy=list(row)
                row_copy[-1]=None
                test_set.append(row_copy)
        predicted=algorithm(dataset,test_set)
        print(predicted)
        actual=[row[-1] for row in dataset]
        rmse=rmse_metric(actual,predicted)
        return rmse



# define simple linear regression 
def simple_linear_regression(train,test):
        predictions=list()
        b0,b1=coefficient(train)
        for row in test:
                yhat=b0+b1*row[0]
                predictions.append(yhat)
        return predictions

# calling
rmse=evaluate_algorithm(dataset,simple_linear_regression)
print('avearge on behalf first vehicle that is four wheelers: %.3f'%(rmse))
print('average fourwheelers in per hour :',66)
print('similar for all others vehicle')


plt.xlabel('fourwheelersRNN ')
plt.ylabel('mean valus using RNN')
plt.scatter(x1,x2)
plt.scatter(x,y)
plt.show()



