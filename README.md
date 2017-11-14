# MachineLearning
>Programming assignments about machine learning from coursera and some interesting exercises

K Nearest Neighbor Algorithm for example:

## Introduction
In this exercise, you will implement k nearest neighbor algorithm with k-dimension tree and get to see it work on data. Before starting on this programming  exercise, we strongly recommend watching the book 《statistical learning method》and understanding what the core concept the algorithm is.

To get started with the exercise, you will need to have some knowledge about python, especially numpy and list, but if you don’t know python at all, it doesn’t matter and you can find instructions on their website to get a quick view.

## What is k nearest neighbor
K nearest neighbor is a based algorithm for classification and regression. In this experiment we only implement k nearest neighbor algorithm for classification. Given a new input sample, then find the k samples which are nearest away from the new input sample in the training set, they are defined as k nearest neighbors, finally we will make decision according to the k nearest neighbors, that is to say if most of the k nearest neighbors are apple, so we will classify the new input sample as apple as well. That’s the core concept of this algorithm as far as I am concerned.

Now we have already know something about what k nearest neighbor is, the next thing we need to know is: 

* **How to choose the value of k.**
* **How to define the nearest.**
* **How to find the k samples or elements.**

For more [details](https://github.com/wangruns/MachineLearning/blob/master/knn-algorithm/knn-algorithm-exercise.md).



***
Linear Regression Algorithm for example:

## Introduction
In this exercise, you will implement linear regression algorithm with gradient descent. Given a new input sample X, and then predict the value of y. 

To get started with the exercise, you will need to have some knowledge about python, especially numpy and DataFrame, but if you don’t know python at all, it doesn’t matter and you can find instructions on their website to get a quick view.

## What is linear regression
As far as I am concerned, Linear regression is a function composed of some single variences whose power is 1. Say that we have n featrues for our sample, so we can write our linear regression hypothesis model as:

h=theta0 + theta1\*x1 + theta2\*x2 + ...thetan\*xn

For more [details](https://github.com/wangruns/MachineLearning/blob/master/linear-regression-algorithm/linear-regression-algorithm-exercise.md). 
