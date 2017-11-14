# Linear Regression Algorithm Exercise
>wangruns

## Introduction
In this exercise, you will implement linear regression algorithm with gradient descent. Given a new input sample X, and then predict the value of y.

To get started with the exercise, you will need to have some knowledge about python, especially numpy and DataFrame, but if you don’t know python at all, it doesn’t matter and you can find instructions on their website to get a quick view.

## What is linear regression
As far as I am concerned, Linear regression is a function composed of some single variences whose power is 1. Say that we have n featrues for our sample, so we can write our linear regression hypothesis model as:

h=theta0 + theta1\*x1 + theta2\*x2 + ...thetan\*xn

Now we have already know something about what linear regression is, the next thing we need to know is: 

* **define linear regression hypothesis model.**
* **find the most suitable parameters that has low cost.**

In this exercise I will use gradient descent to figure the parameters out. You can also choose *Normal Equations*. If you are interested in *Normal Equations*, actually,it's a easy way to help us to calcute the theta just in one calculation without any mean normalizaion and feature scaling.

theta=((X.transpose\*X).inverse)\*X.transpose\*y 


## Define linear regression hypothesis model
Say that the dimensions of feature X is n, so that we have hypothesis()=theta0 + theta1\*x1 + theta2*\x2 + ... + thetan\*xn. More specifically, we have 5 features in our samples so there may be 5-6 parameters to be calculate, say that we have theta0 as a scalar, namely we have 6 parameters for our linear regression hypothesis. e.g

h=theta0 + theta1\*x1 + theta2\*x2 + ... + theta5\*x5

## Gradient descent
Say that our linear regression hypothesis model is:


![image0001](https://raw.githubusercontent.com/wangruns/wangruns.github.io/master/images/blog/linear-regression/Selection_016.png)

and the objective of linear regression is to minimize the cost function:


![image0001](https://raw.githubusercontent.com/wangruns/wangruns.github.io/master/images/blog/machine-learning-ex1/Selection_006.png)

and we will use the gradient descent to update the parameter theta:


![image0001](https://raw.githubusercontent.com/wangruns/wangruns.github.io/master/images/blog/machine-learning-ex1/Selection_004.png)


## Classification exercise
This problem has the following inputs: 

1. Frequency, in Hertzs. 
2. Angle of attack, in degrees. 
3. Chord length, in meters. 
4. Free-stream velocity, in meters per second. 
5. Suction side displacement thickness, in meters.  
6. Scaled sound pressure level, in decibels(the only output). 

The training set looks like as followed:

800,0,0.3048,71.3,0.00266337,126.201

1000,0,0.3048,71.3,0.00266337,125.201

……

and the test set looks like as followed:
1250,0,0.3048,71.3,0.00266337

1600,0,0.3048,71.3,0.00266337

……

Our work is to get our regression model via the training set and use it to predict the scaled sound pressure level on test set. Actually, we can use 5 steps to address the problem.

### Part 1: Data preprocessing.

```python
## ====================Part 1: Data Preprocessing====================
	#  Load the data and transform the form of data to what we need. e.g
	#  vector or matrix
	#training set and test set input
	df=pd.read_table(path+'./noise_train.txt',header=None,sep=',');
	df_test=pd.read_table(path+'./noise_test.txt',header=None,sep=',');

```

### Part 2: Choose data and separate label

```python
## ===============Part 2: Choose data and separate label===============
	#  randomly choose training set and cross validation set, here i set the
	#  ratio to 7:3
	#  then separate the features and label
	#randomly choose training set and cross validation set
	ratio=0.7;
	labelIndex=5;
	n=int(ratio*len(df));
	trainSet=df[:n].sample(n);
	validationSet=df[n:].sample(len(df)-n);
	X_train=trainSet.drop(labelIndex,axis=1).values;
	y_train=trainSet[labelIndex].values;
	X_val=validationSet.drop(labelIndex,axis=1).values;
	y_val=validationSet[labelIndex].values;
	X_test=df_test.values;
	#feature normalize
	X_train_norm=featureNormalize(X_train);
	X_val_norm=featureNormalize(X_val);
	X_test_norm=featureNormalize(X_test);

```

### Part 3: Training

```python
## ====================Part 3: Training====================
	#  notice that we have 5 features in our samples, so there
	#  may be 5-6 parameters to be calculate, say that we have
	#  theta0 as a scalar, namely we have 6 parameters for our
	#  linear regression hypothesis. e.g
	#  h=theta0 + theta1*x1 + theta2*x2 + ... + theta5*x5
	#  as you know, our task is to find the most suitable parameters
	#  theta here to fit our training set and cross validation set
	#  
	theta=linearRegression(X_train_norm,y_train,alpha,iteration);

```

### Part 4: Validation

```python
## ====================Part 4: Validation====================
	#  after we get our hypothesis model via our training set, then
	#  we need to use cross validation to see how well it works for
	#  the new samples.
	#  we can see that the squared error J of all the cross validation
	#  data set is about 21.9643222089
	estimate(theta,X_val_norm,y_val);
	print 'Please any key to begin predicting...';
	sys.stdin.readline();

```

### Part 5: Prediction

```python
## ====================Part 5: Prediction====================
	#  we can use our hypothesis model well trained to help us to 
	#  make decision
	#
	#test
	m=len(X_test_norm);
	X_test_norm=np.c_[np.ones(m),X_test_norm];
	y_pre=X_test_norm.dot(theta);
	cnt=0;
	for i in X_test:
		print i,'predicted as>>',y_pre[cnt];cnt+=1;

```


## Programming codes
```python
## Linear regression algorithm learn and exercise

#  Instructions
#  ------------
#
#  Linear regression is a based and easy algorithm for regression problems
#  it actual can be divided into two steps i think:
#
#  the first step is to define our linear hypothesis function according to 
#  the dimensions of the feature of training set X. e.g
#  say that the dimensions of feature X is n, so that we have
#  hypothesis()=theta0 + theta1*x1 + theta2*x2 + ... + thetan*xn
#  
#  the second step is to find the most suitable parameters theta, which fit
#  our trainig set and cross validation set well. To make this come true, we
#  may have two methods totally, and one of them is gradient descent, another
#  one is Normal Equations, but in this practice i will choose the fisrt one
#  that is gradient descent
#  (if you are interested in Normal Equations, actually,it's a easy way to help us)
#  (to calcute the theta just in one calculation without any mean normalizaion and)
#  (feature scaling. theta=((X.transpose*X).inverse)*X.transpose*y                )
#
#  this was a bit easy practice compared with the last one i think...
#  wangurns
#  2017-11-13 20:57:04
#


#coding:utf8

import sys
import numpy as np
import pandas as pd

answer = []
trainSet = []
testSet = []
#the path of train.txt and test.txt files, here current path is used
path='';
#the number of iterations of gradient descent
iteration=500;
#the learning rate for each iteration
alpha=0.03;

def estimate(theta,X_val,y_val):
	m=len(y_val);
	X=np.c_[np.ones(m),X_val];
	y_pre=X.dot(theta);
	error=y_pre-y_val;
	J=1./(2*m)*sum(error**2);
	print 'the squared error J for validation set is',float(J);

def featureNormalize(X_train):
	mu=X_train.mean(axis=0);
	sigma=np.std(X_train,axis=0);
	X_norm=(X_train-mu)/sigma;
	return X_norm;

def linearRegression(X,y,alpha,iteration):
	m=len(y);
	X=np.c_[np.ones(m),X];
	theta=np.random.rand(len(X[0]));
	for i in range(iteration):
		#gradientDescent
		error=X.dot(theta)-y;
		"""
		#the cost J is reducing after each iteration totally
		J=1./(2*m)*sum(error**2);
		print 'the squared cost J is',float(J);
		"""
		theta-=alpha/m*(X.T.dot(error));
	return theta;

def train():
	## ====================Part 1: Data Preprocessing====================
	#  Load the data and transform the form of data to what we need. e.g
	#  vector or matrix
	#training set and test set input
	df=pd.read_table(path+'./noise_train.txt',header=None,sep=',');
	df_test=pd.read_table(path+'./noise_test.txt',header=None,sep=',');


	## ===============Part 2: Choose data and separate label===============
	#  randomly choose training set and cross validation set, here i set the
	#  ratio to 7:3
	#  then separate the features and label
	#randomly choose training set and cross validation set
	ratio=0.7;
	labelIndex=5;
	n=int(ratio*len(df));
	trainSet=df[:n].sample(n);
	validationSet=df[n:].sample(len(df)-n);
	X_train=trainSet.drop(labelIndex,axis=1).values;
	y_train=trainSet[labelIndex].values;
	X_val=validationSet.drop(labelIndex,axis=1).values;
	y_val=validationSet[labelIndex].values;
	X_test=df_test.values;
	#feature normalize
	X_train_norm=featureNormalize(X_train);
	X_val_norm=featureNormalize(X_val);
	X_test_norm=featureNormalize(X_test);
	

	## ====================Part 3: Training====================
	#  notice that we have 5 features in our samples, so there
	#  may be 5-6 parameters to be calculate, say that we have
	#  theta0 as a scalar, namely we have 6 parameters for our
	#  linear regression hypothesis. e.g
	#  h=theta0 + theta1*x1 + theta2*x2 + ... + theta5*x5
	#  as you know, our task is to find the most suitable parameters
	#  theta here to fit our training set and cross validation set
	#  
	theta=linearRegression(X_train_norm,y_train,alpha,iteration);
	
	
	## ====================Part 4: Validation====================
	#  after we get our hypothesis model via our training set, then
	#  we need to use cross validation to see how well it works for
	#  the new samples.
	#  we can see that the squared error J of all the cross validation
	#  data set is about 21.9643222089
	estimate(theta,X_val_norm,y_val);
	print 'Please any key to begin predicting...';
	sys.stdin.readline();


	## ====================Part 5: Prediction====================
	#  we can use our hypothesis model well trained to help us to 
	#  make decision
	#
	#test
	m=len(X_test_norm);
	X_test_norm=np.c_[np.ones(m),X_test_norm];
	y_pre=X_test_norm.dot(theta);
	cnt=0;
	for i in X_test:
		print i,'predicted as>>',y_pre[cnt];cnt+=1;
	
	
if __name__ == '__main__':
	train()

```