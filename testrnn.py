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

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
 
# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))
     
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
 
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])



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
#x1=[irrr[2] for irrr in  z]
#x2=[irrrr[3] for irrrr in  z]
print(x)
print("_____________________")
print(y)
# mean x and mean y
mean_x,mean_y=mean(x),mean(y)
var_x,var_y=variance(x,mean_x),variance(y,mean_y)
'''
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
    return [o, s]

# calling  constructor  
RNNNumpy.forward_propagation = forward_propagation


#  prediction process

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)
 
RNNNumpy.predict = predict


#  testing  

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(x[10])
print(o.shape)
