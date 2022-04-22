# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:49:43 2020

@author: user
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('CSVs//Input n Capacity.csv')
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

skf = KFold(n_splits=5, shuffle=True)
fold_no = 0
traintot = 0
testtot = 0
for train_index,test_index in skf.split(dataset, y):
    train = dataset.iloc[train_index,:]
    test = dataset.iloc[test_index,:]

    Xtrain = train.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytrain = train["Capacity(Ah)"]

    Xtest = test.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytest = test["Capacity(Ah)"]

    lin_reg = LinearRegression()
    lin_reg.fit(Xtrain,Ytrain)
 
    ytrain_pred = lin_reg.predict(Xtrain)
    trainscore = r2_score(Ytrain, ytrain_pred)
    ytest_pred = lin_reg.predict(Xtest)
    testscore = r2_score(Ytest, ytest_pred)

    traintot += trainscore
    testtot += testscore
    fold_no += 1

    plt.scatter(Xtrain.iloc[:,0], Ytrain, color = 'red')
    plt.plot(Xtest.iloc[:, 0], lin_reg.predict(Xtest), color = 'blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Cycle')
    plt.ylabel('Capacity')
    plt.show()

    print("Linear Regression\n")
    print('Fold no',fold_no,'\n')
    print('Number of training examples',len(train.index),'\n')
    print('Number of testing examples',len(test.index),'\n')
    print(pd.DataFrame({'Actual':Ytest,'Predicted':np.ravel(ytest_pred)}).iloc[:10].to_string(index=False))
    print('Training accuracy:', trainscore)
    print('Testing accuracy:', testscore)
   
print('Average training accuracy:', traintot/fold_no)
print('Average testing accuracy:', testtot/fold_no)

skf = KFold(n_splits=5, shuffle=True)
fold_no = 0
traintot = 0
testtot = 0
for train_index,test_index in skf.split(dataset, y):
    train = dataset.iloc[train_index,:]
    test = dataset.iloc[test_index,:]

    Xtrain = train.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytrain = train["Capacity(Ah)"]

    Xtest = test.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytest = test["Capacity(Ah)"]

    print("coming here: 81")

    
    poly_reg = PolynomialFeatures(degree = 10)
    X_poly = poly_reg.fit_transform(Xtrain)
    poly_reg.fit(X_poly, Ytrain)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, Ytrain)
    print("coming here: 87")
    ytrain_pred = lin_reg_2.predict(poly_reg.fit_transform(Xtrain))
    trainscore = r2_score(Ytrain, ytrain_pred)
    ytest_pred = lin_reg_2.predict(poly_reg.fit_transform(Xtest))
    testscore = r2_score(Ytest, ytest_pred)
    print("coming here: 92")
    traintot += trainscore
    testtot += testscore
    fold_no += 1

    plt.scatter(Xtrain.iloc[:,0], Ytrain, color = 'red')
    plt.plot(Xtest.iloc[:, 0], lin_reg.predict(Xtest), color = 'blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Cycle')
    plt.ylabel('Capacity')
    plt.show()

    print("Polynomial Regression\n")
    print('Fold no',fold_no,'\n')
    print('Number of training examples',len(train.index),'\n')
    print('Number of testing examples',len(test.index),'\n')
    print(pd.DataFrame({'Actual':Ytest,'Predicted':np.ravel(ytest_pred)}).iloc[:10].to_string(index=False))
    print('Training accuracy:', trainscore)
    print('Testing accuracy:', testscore)
   
print('Average training accuracy:', traintot/fold_no)

print('Average testing accuracy:', testtot/fold_no)

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)"""

# # Fitting Linear Regression to the dataset
# from sklearn.linear_model import LinearRegression, Ridge
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)

# # Fitting Polynomial Regression to the dataset
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = 10)
# X_poly = poly_reg.fit_transform(X_train)
# poly_reg.fit(X_poly, y_train)
# # lin_reg_2 = LinearRegression()
# lin_reg_2 = Ridge()

# lin_reg_2.fit(X_poly, y_train)

# # Visualising the Linear Regression results
# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color = 'blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Cycle')
# plt.ylabel('Capacity')
# plt.show()

# # Visualising the Polynomial Regression results
# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# # Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# # Predicting a new result with Linear Regression
# #lin_reg.predict(6.5)

# # Predicting a new result with Polynomial Regression
# y_pred= lin_reg_2.predict(poly_reg.fit_transform(X_test))

# #Accuracy

# import sklearn
# import math
# R2= r2_score(y_test, y_pred)
# mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
# rmse = math.sqrt(mse)

# import math
# numsum=0
# densum=0
# mean=0
# summ=0
# for i in range(0,212):
#     summ=summ+y_test[i]
    
# mean=summ/128

# for i in range(0,212):
#     numsum=numsum+(y_test[i]-y_pred[i])**2
#     densum=densum+(y_test[i]-mean)**2
    
# R2=1-(numsum/densum)
# #import math
# Rmse = math.sqrt(numsum/212)