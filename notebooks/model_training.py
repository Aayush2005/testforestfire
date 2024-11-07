import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Algerian_forest_fires_cleaned_dataset.csv')
df.drop(['month','day','year'],axis=1,inplace=True)
df['Classes'] = np.where(df['Classes'].str.contains('not fire'),0,1)
print(df.head())

print(df['Classes'].value_counts())

#Independent and dependent feature

X = df.drop('FWI',axis = 1)
y = df['FWI']

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#Feature selection on the basis of correlation

# print(X_train.corr())

#Check for multicollinearity
# plt.figure(figsize=(12,10))
# corr = X_train.corr()
# sns.heatmap(corr,annot=True)

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


#Setting threshhold is done by domain expert
cols = correlation(X_train,0.85)
print(cols)

#Drop these features
X_train.drop(cols,axis=1,inplace=True)
X_test.drop(cols,axis=1,inplace=True)


#Standardization of data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Box plot to understand the effect of standard scaler

##Applying Linear regression model

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X_train_scaled,y_train)
y_pred = linreg.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error,r2_score
mae = mean_absolute_error(y_test,y_pred)

score = r2_score(y_test,y_pred)
print("Mean Absolute error",mae)
print("Score",score)

#Scaller plot
# plt.scatter(y_test,y_pred)
# plt.show()

#Cross validation Lasso
from sklearn.linear_model import LassoCV
lassoCV = LassoCV(cv=5)
lassoCV.fit(X_train_scaled,y_train)
print(lassoCV.alpha_)
#Applying lasso regression

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=lassoCV.alpha_)

lasso.fit(X_train_scaled,y_train)
y_pred = lasso.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error,r2_score
mae = mean_absolute_error(y_test,y_pred)

score = r2_score(y_test,y_pred)
print("Mean Absolute error",mae)
print("Score",score)

#Scatter plot
# plt.scatter(y_test,y_pred)
# plt.show()

#RidgeCV

#Ridge regression
print("Ridge)")

from sklearn.linear_model import Ridge
ridge = Ridge()

ridge.fit(X_train_scaled,y_train)
y_pred = ridge.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error,r2_score
mae = mean_absolute_error(y_test,y_pred)

score = r2_score(y_test,y_pred)
print("Mean Absolute error",mae)
print("Score",score)

#Scaller plot
# plt.scatter(y_test,y_pred)
# plt.show()

#ELastic net CV

#Elastic Net


from sklearn.linear_model import ElasticNet
enet = ElasticNet()

enet.fit(X_train_scaled,y_train)
y_pred = enet.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error,r2_score
mae = mean_absolute_error(y_test,y_pred)

score = r2_score(y_test,y_pred)
print("Mean Absolute error",mae)
print("Score",score)

#Scaller plot
# plt.scatter(y_test,y_pred)
# plt.show()

import pickle
pickle.dump(scaler,open('Ridge_scaler.pkl','wb'))
pickle.dump(ridge,open('ridge.pkl','wb'))