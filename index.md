## DATA 310 Midterm Project

### Question 1
Import the weatherHistory.csv into a data frame. How many observations do we have? 

```markdown
import pandas as pd
df = pd.read_csv('weatherHistory.csv')
df.shape
```

Answer: 96,453

### Question 2
In the weatherHistory.csv data how many features are just nominal variables?

Answer: 3

### Question 3
If we want to use all the unstandardized observations for 'Temperature (C)' and predict the Humidity the resulting root mean squared error is (just copy the first 4 decimal places):

```markdown
X = df['Temperature (C)'].values
y = df['Humidity'].values

model = LinearRegression()
model.fit(X.reshape(-1,1), y) 
y_pred = model.predict(X.reshape(-1,1)) # predict X or y?

rmse = np.sqrt(mean_squared_error(y,y_pred))
print(rmse)
```
Answer: 0.1514

### Question 4
If the input feature is the Temperature and the target is the Humidity and we consider 20-fold cross validations with random_state=2020, the Ridge model with alpha =0.1 and standardize the input train and the input test data. The average RMSE on the test sets is (provide your answer with the first 6 decimal places):

```markdown
from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=20, random_state=2020,shuffle=True)

X = df['Temperature (C)'].values
y = df['Humidity'].values

scale = StandardScaler() 
PE = []
PE_train = []
model = Ridge(alpha = 0.1)
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_train_scaled = scale.fit_transform(X_train.reshape(-1,1))
    y_train = y[train_index]
    X_test = X[test_index]
    X_test_scaled = scale.transform(X_test.reshape(-1,1)) # use fit_transform or transform?
    y_test = y[test_index]
    model = model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)
    PE_train.append(np.sqrt(MSE(y_train, y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))
    # print('RMSE from each fold:',np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated prediction error for train set is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated prediction error for test set is: ' + str(np.mean(PE)))
```
Answer:  0.151438

### Question 5
Suppose we want to use Random Forrest with 100 trees and max_depth=50 to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-cross validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 6 decimal places):

```markdown
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def RMSE(y,yhat):
    return np.sqrt(MSE(y,yhat))
    
def DoKFold(X,y,model,k, rs):
    PE1 = []
    PE2 = []
    kf = KFold(n_splits=k,shuffle=True,random_state=1693)
    for idxtrain,idxtest in kf.split(X):
        Xtrain = X[idxtrain,:]
        Xtest  = X[idxtest,:]
        ytrain = y[idxtrain]
        ytest  = y[idxtest]
        model.fit(Xtrain,ytrain)
        yhat = model.predict(Xtest)
        #PE1.append(MAEf(ytest,yhat))
        PE2.append(RMSE(ytest,yhat))
    return np.mean(PE2)
    
X = df['Apparent Temperature (C)'].values.reshape(-1,1)
y = df['Humidity'].values

model = RandomForestRegressor(n_estimators=100,max_depth=50)
DoKFold(X, y, model, 10, 1693)
```
Answer: 0.143502

### Question 6
Suppose we want use polynomial features of degree 6 and we want to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 5 decimal places):
```markdown
X = df['Apparent Temperature (C)'].values
y = df['Humidity'].values

polynomial_features= PolynomialFeatures(degree=6)
x_poly = polynomial_features.fit_transform(X.reshape(-1,1))

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
print(rmse)

DoKFold(x_poly, y, model, 10, 1693)
```
Answer:  0.14346

### Question 7
If the input feature is the Temperature and the target is the Humidity and we consider 10-fold cross validations with random_state=1234, the Ridge model with alpha =0.2. Inside the cross-validation loop standardize the input data. The average RMSE on the test sets is (provide your answer with the first 4 decimal places):
```markdown
X = df['Temperature (C)'].values
y = df['Humidity'].values

kf = KFold(n_splits=10, random_state=1234,shuffle=True)

PE = []
PE_train = []
model = Ridge(alpha = 0.2)
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_train_scaled = scale.fit_transform(X_train.reshape(-1,1))
    y_train = y[train_index]
    X_test = X[test_index]
    X_test_scaled = scale.transform(X_test.reshape(-1,1)) # use fit_transform or transform?
    y_test = y[test_index]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)
    PE_train.append(np.sqrt(MSE(y_train, y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))
    # print('RMSE from each fold:',np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated prediction error for train set is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated prediction error for test set is: ' + str(np.mean(PE)))
```
Answer:  7.4001

### Question 8
Suppose we use polynomial features of degree 6 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 4 decimal places):
```markdown
X = df[['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']].values
y = df['Temperature (C)'].values

polynomial_features= PolynomialFeatures(degree=6)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()

DoKFold(x_poly, y, model, 10, 1234)
```
Answer: 6.0776

### Question 9
Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 4 decimal places):
```markdown
X = df[['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']].values
y = df['Temperature (C)'].values

model = RandomForestRegressor(n_estimators=100,max_depth=50)
DoKFold(X,y,model,10,1234)
```
Answer: 5.8335

### Question 10
If we visualize a scatter plot for Temperature (on the horizontal axis) vs Humidity (on the vertical axis) the overall trend seems to be 

Answer: Decreasing
