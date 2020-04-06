import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures

import \
    pandas as pd


import numpy as np

print("NumPy version: {}".format(np.__version__))

import scipy as sp

print("SciPy version: {}".format(sp.__version__))

import IPython

print("IPython version: {}".format(IPython.__version__))

import sklearn

print("scikit-learn version: {}".format(sklearn.__version__))
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', 100)

# misc libraries
import random
import time

import warnings

warnings.filterwarnings('ignore')
print('-' * 25)

data_raw = pd.read_csv(r"C:\Users\augus\Desktop\train.csv")

to_drop=['Street','Alley','Utilities','Condition2','RoofMatl','MasVnrArea','Heating','CentralAir','LowQualFinSF','3SsnPorch','LotFrontage','PoolArea','PoolQC','MiscFeature','MiscVal']


data_val  = pd.read_csv(r"C:\Users\augus\Desktop\test.csv")
data1 = data_raw.copy(deep = True)
data_cleaner = [data1, data_val]


for dataset in data_cleaner:
   dataset.drop(to_drop, axis=1, inplace=True)


for dataset in data_cleaner:
    dataset['BsmtQual'].fillna('NoBsmt', inplace=True)
    dataset['BsmtCond'].fillna('NoBsmt', inplace=True)
    dataset['BsmtExposure'].fillna('NoBsmt', inplace=True)
    dataset['BsmtFinType1'].fillna('NoBsmt', inplace=True)
    dataset['BsmtFinType2'].fillna('NoBsmt', inplace=True)
    dataset['Electrical'].fillna(dataset['Electrical'].mode()[0],inplace=True)
    dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode()[0],inplace=True)
    dataset['FireplaceQu'].fillna('NoFireplace', inplace=True)
    dataset['GarageType'].fillna('NoGarage', inplace=True)
    dataset['GarageYrBlt'].fillna(0, inplace=True)
    dataset['GarageFinish'].fillna('NoGarage', inplace=True)
    dataset['GarageQual'].fillna('NoGarage', inplace=True)
    dataset['GarageCond'].fillna('NoGarage', inplace=True)
    dataset['Fence'].fillna(dataset['Fence'].mode()[0], inplace=True)
    dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0],inplace=True)
    dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0],inplace=True)
    dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0],inplace=True)
    dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0],inplace=True)
    dataset['Functional'].fillna(dataset['Functional'].mode()[0],inplace=True)
    dataset['SaleType'].fillna(dataset['SaleType'].mode()[0],inplace=True)
    dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode()[0],inplace=True)
    dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mode()[0],inplace=True)
    dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mode()[0],inplace=True)
    dataset['TotalBsmtSF'].fillna(0,inplace=True)
    dataset['BsmtFullBath'].fillna(0,inplace=True)
    dataset['BsmtHalfBath'].fillna(0,inplace=True)
    dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean(),inplace=True)
    dataset['GarageCars'].fillna(0,inplace=True)
    dataset['GarageArea'].fillna(0,inplace=True)


#print(data_val.isna().sum())
""""
print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)
"""





#one hot encode categorical data
categorical_data=['LotShape','MSZoning','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','HeatingQC','Electrical','2ndFlrSF','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition','Fence']
for x in categorical_data:
    data_cleaner=[data1,data_val]
   # print("Au tour de " + x)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(dataset[[x]]).toarray())
    enc_df.columns = enc.get_feature_names([x])
   # print(enc_df.columns)
    for dataset in data_cleaner:
        dataset.drop([x],axis=1,inplace=True)
      #  print("On supprime " + x + "de " + str(type(dataset)))
      #  print(dataset)
        if (dataset.equals(data1)):
            data1 = dataset.join(enc_df)
        else:
            data_val=dataset.join(enc_df)



Target = ['SalePrice']


data1.drop(1459,axis=0,inplace=True)



data1y=data1['SalePrice']
data1.drop(['SalePrice'],axis=1,inplace=True)

poly = PolynomialFeatures(degree=2)
data1 = poly.fit_transform(data1)

X_train, X_test, y_train, y_test = train_test_split(data1, data1y,random_state = 0)

print(X_train)
print(X_test)
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#data1_x_scaled=scaler.fit_transform(data1.drop(['SalePrice'],axis=1,inplace=False))
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=200.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')
""""
data_val_scaled=scaler.transform(data_val)

data_val['SalePrice'] = linlasso.predict(data_val_scaled)


submit = data_val[['Id','SalePrice']]
submit.to_csv("submit.csv", index=False)
"""

