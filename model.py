# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:06:19 2020
@author: Owner
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge, ElasticNet, LassoCV, LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pickle
import requests
import json
import math

housing_df = pd.read_csv('D:/Reverse Engineering/House Price/train.csv')
# housing_df = housing_df.drop(['Id'], axis=1) ## Removing the ID column as it'll have no impact on the predictions

## Imputing Null Values
def check_Null_values():
    housing_df_Null = housing_df.isnull().sum().reset_index()
    housing_df_Null.columns = ['Name', 'Count']
    housing_df_Null = housing_df_Null.loc[housing_df_Null['Count']>0].sort_values(by = ['Count'], ascending = False)
    housing_df_Null['Pecentage'] = round((housing_df_Null['Count']/housing_df.shape[0])*100, 2)
    return housing_df_Null

def outlier_detect(data, var):
    q1 = data[var].quantile(0.25)
    q3 = data[var].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = data.loc[(data['SalePrice'] < fence_low) | (data['SalePrice'] > fence_high)]
    return df_out['Id']

def ordinaltoint_Ex_Po(rating):
    rating_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'none': 0}    
    return(rating_dict[rating])
 
def ordinaltoint_GLQ_Unf(rating):
        rating_dict = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf': 1, 'none':0}
        return(rating_dict[rating])

To_Be_removed = list() ## Will populate this list with the columns to be removed from the database

## Columns with more than half of the records missing, So removing columns wit more than 50% Null rows
To_Be_removed = check_Null_values().loc[check_Null_values()['Pecentage']>50]['Name']

## LotFrontage - Will impute this value with the neighborhood mean
housing_df['LotFrontage'] = housing_df.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

## Garage
for col in ['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']:
    if housing_df[col].dtype == 'O':
        housing_df[col].fillna('none', inplace = True)
    else:
        housing_df[col].fillna(0, inplace = True)
## MasVnrArea
housing_df['MasVnrArea'].fillna(0, inplace=True)
## FireplaceQu
housing_df['FireplaceQu'].fillna('none', inplace=True)
## BsmtExposure
housing_df['BsmtExposure'].fillna('none', inplace=True)
## BsmtFinType2 and BsmtFinType1
housing_df['BsmtFinType1'].fillna('none', inplace=True)
housing_df['BsmtFinType2'].fillna('none', inplace=True)
#BsmtQual
housing_df['BsmtQual'].fillna('none', inplace=True)
#BsmtCond
housing_df['BsmtCond'].fillna('none', inplace=True)
#MasVnrType
housing_df['MasVnrType'].fillna('none', inplace=True)
#Electrical
housing_df['Electrical'].fillna(housing_df['Electrical'].mode()[0], inplace = True)
#PoolQc
housing_df['PoolQC'].fillna("none", inplace=True)
##Checking the NUll Values after null value imputation 

housing_df_Null = check_Null_values()
## Removing all the columns which may lead to overfitting.
## These are the columns with a single value spanning for 99% of the rows leading to low variations for prediction
overfitting = []
for col in housing_df.columns:
    counts = housing_df[col].value_counts().reset_index().loc[0][col]
    perccounts = (counts/len(housing_df[col]))*100
    if (perccounts > 99):
        To_Be_removed = np.append(To_Be_removed, col)

## feature Engineering : We'll create some additional features from already present features in datset. 

housing_df['TotalBathRooms'] = housing_df['BsmtFullBath']+housing_df['FullBath']+(0.5*housing_df['BsmtHalfBath'])+(0.5*housing_df['HalfBath'])
housing_df['TotalPorchArea'] = housing_df['OpenPorchSF']+ housing_df['EnclosedPorch']+housing_df['ScreenPorch']
housing_df['YearsOld'] = (housing_df['YrSold'] - housing_df['YearRemodAdd'])
housing_df['YearsOld'] = housing_df['YearsOld'].apply(lambda x: (0 if (x < 0) else x))
housing_df['TotalArea'] = housing_df['GrLivArea'] + housing_df['TotalBsmtSF'] + housing_df['1stFlrSF'] + housing_df['2ndFlrSF']
housing_df['HasBasement'] = housing_df['TotalBsmtSF'].apply(lambda x: (1 if (x > 0) else 0))

## converting MSSubClass to categorical
housing_df = housing_df.replace({"MSSubClass":{20:"20MS", 30:"30MS", 40:"40MS", 45:"45MS", 50:"50MS", 60:"60MS", 70:"70MS", 75:"75MS", 80:"80MS", 85:"85MS", 90:"90MS", 120:"120MS", 160:"160MS", 180:"180MS", 190:"190MS"}})

## We'll deal with Numeric variables --> ordinal --> Nominal
## Extracting Numeric Variables 

## Calculating the corelation of these variables with the SalePrice (Target Variable)
## We'll only keep variables which have a significant amount of corelation (0.4 or more) with the SalePrice. 

housing_df_Numeric = housing_df.dtypes.reset_index()
housing_df_Numeric.columns = ['Name', 'Dtype']
housing_df_Numeric = housing_df_Numeric.loc[housing_df_Numeric['Dtype']!='object']
housing_df_Numeric = housing_df[housing_df_Numeric['Name']]
housing_df_Numeric.columns

cor = []
for col in housing_df_Numeric:
    cor.append(abs(np.corrcoef(housing_df_Numeric[col] , housing_df_Numeric['SalePrice'])[0,1]))
Corrdf =  pd.DataFrame(cor, housing_df_Numeric.columns).reset_index()
Corrdf.columns= ['Column', 'Corelation']
Corrdf = Corrdf.loc[Corrdf['Corelation']<0.4]
Corrdf.sort_values(by=['Corelation'], ascending=False)
To_Be_removed = np.append(To_Be_removed, Corrdf['Column']) 

## From the selected variables we'll remove the outliers
## Before that we'll remove collinear columns to avoid multicollinearty. 

## Some colinear variables I could find are:
# GrLivArea, 1stFlrSF, TotalArea, TotRmsAbvGrd
# GarageArea, GarageCars
# YearRemodAdd, Yearbuild, YearsOld
# TotalBathRooms, FullBath

# We'll check these and remove some of the features as requitred'

# Total Area is highly corelated with GrlivArea, 1stFlrSF, TotalBsmtSF and TotrmsAbvGrd. We'll only be keeping TotalArea as it has the highest corelation with Sales Price.
To_Be_removed = np.append(To_Be_removed, ['GrLivArea', '1stFlrSF', 'TotRmsAbvGrd', 'TotalBsmtSF'])

# We'll keep GarageCars as it has a higher corelation with SalePrice
To_Be_removed = np.append(To_Be_removed, ['GarageArea'])

# We'll remove YearBuilt and YearRemodAdd and keep Yearsold.
To_Be_removed = np.append(To_Be_removed, ['YearRemodAdd', 'YearBuilt'])

# We'll remove 'FullBath' as TotalBathRooms has a higher corelation with SalePrice
To_Be_removed = np.append(To_Be_removed, ['FullBath'])

## Numeric features selected so far. 
# ## overwriting housing_df_Numeric
Tempdf = housing_df.drop(To_Be_removed, axis=1)
housing_df_Numeric = Tempdf.dtypes.reset_index()
housing_df_Numeric.columns = ['Name', 'Dtype']
housing_df_Numeric = housing_df_Numeric.loc[housing_df_Numeric['Dtype']!='object']
housing_df_Numeric = Tempdf[housing_df_Numeric['Name']]

## Our Target variable is highly skewed. Lets fix all the skewd features. 
housing_df['SalePrice'] = np.log1p(housing_df['SalePrice'])
housing_df['TotalArea'] = np.log1p(housing_df['TotalArea'])

## Exploratory Analysis ##
## OverallQual
data = housing_df[['OverallQual', 'SalePrice']]
f, ax = plt.subplots(figsize=(7, 7))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

## There are a few outliers. Lets see if we can detect their IDs

outlier_Id = []
for valu in pd.unique(housing_df['OverallQual']):
    data = housing_df.loc[housing_df['OverallQual']==valu]
    outlier_Id =np.append(outlier_Id, outlier_detect(data, 'SalePrice'))

## TotalArea 
data = housing_df[['TotalArea', 'SalePrice']]
f, ax = plt.subplots(figsize=(7, 7))
fig = sns.scatterplot(x='TotalArea', y="SalePrice", data=data)

## 2 Houses at the right side of the graph seem to be outliers. 
filt1 = housing_df['TotalArea'] > 9.5
filt2 = housing_df['SalePrice']<14
outlier_Id = np.append(outlier_Id, housing_df.loc[filt1 & filt2]['Id'])

## Garagecars
data = housing_df[['GarageCars', 'SalePrice']]
f, ax = plt.subplots(figsize=(7, 7))
fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)

## TotalBathRooms
data = housing_df[['TotalBathRooms', 'SalePrice']]
f, ax = plt.subplots(figsize=(7, 7))
fig = sns.boxplot(x='TotalBathRooms', y="SalePrice", data=data)

## YearsOld
Data = housing_df[['YearsOld', 'SalePrice']]
sns.scatterplot(x='YearsOld', y='SalePrice', data=Data)

## MasVnrArea
# But there are 869 of these cases with MasVnrArea as 0. 
# So its better to drop this variable. 
To_Be_removed = np.append(To_Be_removed, ['MasVnrArea'])

## Skewness has already been handled. 
# housing_df_numeric = housing_df.select_dtypes(exclude = 'object').apply(lambda x: skew(x)).sort_values(ascending = False)
# high_skew = housing_df_numeric.loc[abs(housing_df_numeric) >= 0.5]

################# ordinal variables #################
arrord = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu','GarageQual', 'GarageCond', 'PoolQC']
arrord2 = ['BsmtFinType1', 'BsmtFinType2']
rating_dict = {}

for col in arrord:
    housing_df[col] = housing_df[col].apply(lambda x:ordinaltoint_Ex_Po(x))
    
for col in arrord2:
    housing_df[col] = housing_df[col].apply(lambda x:ordinaltoint_GLQ_Unf(x))
    
# LotShape
rating_dict = {"Reg": 1, "IR1": 0, "IR2": 0, "IR3": 0}
housing_df['LotShape'] = housing_df['LotShape'].apply(lambda x: rating_dict[x])
rating_dict = {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
housing_df['Utilities'] = housing_df['Utilities'].apply(lambda x: rating_dict[x])
rating_dict = {"Gtl": 1, "Mod":2, "Sev":3}
housing_df['LandSlope'] = housing_df['LandSlope'].apply(lambda x: rating_dict[x])
rating_dict = {"Gd":4, "Av": 3, "Mn": 2, "No": 1, "none": 0}
housing_df['BsmtExposure'] = housing_df['BsmtExposure'].apply(lambda x: rating_dict[x])
rating_dict = {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0, "none":0}
housing_df['GarageFinish'] = housing_df['GarageFinish'].apply(lambda x: rating_dict[x])
rating_dict = {"Y": 3, "P": 2, "N":1}
housing_df['PavedDrive'] = housing_df['PavedDrive'].apply(lambda x: rating_dict[x])

#################### This part needs to be uncommented  ########################

# There are a few variables which can be combined togeather. These are :

# GarageCond and GarageQual
# BsmtFinType1 and BsmtFinType2
# BsmtQual and BsmtCond
# ExterQual and ExterCond

# temp = housing_df.loc[:, ["GarageCond","GarageQual"]]
# housing_df['GarageNew'] = temp.mean(axis=1)
# To_Be_removed = np.append(To_Be_removed, ["GarageCond","GarageQual"])

# temp = housing_df.loc[:, ["BsmtFinType1","BsmtFinType2"]]
# housing_df['BsmtFinTypeNew'] = temp.mean(axis=1)
# To_Be_removed = np.append(To_Be_removed, ["GarageCond","GarageQual"])

# temp = housing_df.loc[:, ["BsmtQual","BsmtCond"]]
# housing_df['Bsmtnew'] = temp.mean(axis=1)
# To_Be_removed = np.append(To_Be_removed, ["BsmtQual","BsmtCond"])

# temp = housing_df.loc[:, ["ExterQual","ExterCond"]]
# housing_df['Externew'] = temp.mean(axis=1)
# To_Be_removed = np.append(To_Be_removed, ["ExterQual","ExterCond"])

####################################################################################

housing_df_Ordinal = housing_df[["LotShape","Utilities","LandSlope","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","KitchenQual","FireplaceQu","GarageFinish","GarageQual","GarageCond","PavedDrive", "SalePrice"]]
cor = []
for col in housing_df_Ordinal.columns:
    cor.append(abs(np.corrcoef(housing_df_Ordinal[col] , housing_df_Ordinal['SalePrice'])[0,1]))

## Like before I'll be adding the columns with low corelation with SalePrice to 'To_be_removed array'
Corrdf =  pd.DataFrame(cor, housing_df_Ordinal.columns).reset_index()
Corrdf.columns= ['Column', 'Corelation']
Corrdf.sort_values(by = ['Corelation'], ascending = False)
filt1 = Corrdf['Corelation'] < 0.4
To_Be_removed = np.append(To_Be_removed, Corrdf.loc[filt1]['Column'])

## Nominal Variables
## There are a few variables which can be merged togeather. 
## Condition 1 and Condition 2
## Exterior1st and Exterior2nd

housing_df.loc[(housing_df['Condition1'] == 'Artery') | (housing_df['Condition2'] == 'Artery'), 'Artery'] = 1
housing_df.loc[(housing_df['Condition1'] == 'Feedr') | (housing_df['Condition2'] == 'Feedr'), 'Feedr'] = 1
housing_df.loc[(housing_df['Condition1'] == 'Norm') | (housing_df['Condition2'] == 'Norm'), 'Norm'] = 1
housing_df.loc[(housing_df['Condition1'] == 'PosA') | (housing_df['Condition2'] == 'PosA'), 'PosA'] = 1
housing_df.loc[(housing_df['Condition1'] == 'PosN') | (housing_df['Condition2'] == 'PosN'), 'PosN'] = 1
housing_df.loc[(housing_df['Condition1'] == 'RRAe') | (housing_df['Condition2'] == 'RRAe'), 'RRAe'] = 1
housing_df.loc[(housing_df['Condition1'] == 'RRAn') | (housing_df['Condition2'] == 'RRAn'), 'RRAn'] = 1
housing_df.loc[(housing_df['Condition1'] == 'RRNe') | (housing_df['Condition2'] == 'RRNe'), 'RRNe'] = 1
housing_df.loc[(housing_df['Condition1'] == 'RRNn') | (housing_df['Condition2'] == 'RRNn'), 'RRNn'] = 1

for col in ["Artery","Feedr","Norm","PosA","PosN","RRAe","RRAn","RRNe","RRNn"]:
    housing_df[col].fillna(0, inplace = True)
    housing_df[col].astype(int)
    
housing_df.loc[(housing_df['Exterior1st'] == 'VinylSd') | (housing_df['Exterior2nd'] == 'VinylSd'), 'VinylSd'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'MetalSd') | (housing_df['Exterior2nd'] == 'MetalSd'), 'MetalSd'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'Wd Sdng') | (housing_df['Exterior2nd'] == 'Wd Sdng'), 'Wd Sdng'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'HdBoard') | (housing_df['Exterior2nd'] == 'HdBoard'), 'HdBoard'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'BrkFace') | (housing_df['Exterior2nd'] == 'BrkFace'), 'BrkFace'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'WdShing') | (housing_df['Exterior2nd'] == 'WdShing'), 'WdShing'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'CemntBd') | (housing_df['Exterior2nd'] == 'CemntBd'), 'CemntBd'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'Plywood') | (housing_df['Exterior2nd'] == 'Plywood'), 'Plywood'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'AsbShng') | (housing_df['Exterior2nd'] == 'AsbShng'), 'AsbShng'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'Stucco') | (housing_df['Exterior2nd'] == 'Stucco'), 'Stucco'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'BrkComm') | (housing_df['Exterior2nd'] == 'BrkComm'), 'BrkComm'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'AsphShn') | (housing_df['Exterior2nd'] == 'AsphShn'), 'AsphShn'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'Stone') | (housing_df['Exterior2nd'] == 'Stone'), 'Stone'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'ImStucc') | (housing_df['Exterior2nd'] == 'ImStucc'), 'ImStucc'] = 1
housing_df.loc[(housing_df['Exterior1st'] == 'CBlock') | (housing_df['Exterior2nd'] == 'CBlock'), 'CBlock'] = 1
    
for col in ['VinylSd','MetalSd','Wd Sdng','HdBoard','BrkFace','WdShing','CemntBd','Plywood','AsbShng','Stucco','BrkComm','AsphShn','Stone','ImStucc','CBlock']:
    housing_df[col].fillna(0, inplace = True)
    housing_df[col].astype(int)

##Removing Condition 1,Condition 2, Exterior1st and Exterior2nd
housing_df.drop(['Condition1','Condition2','Exterior1st','Exterior2nd'], axis=1, inplace=True)
    
## We'll use one hot encoding to convert nominal variables 
housing_df_nominal = housing_df.select_dtypes(include = 'object').columns
housing_df_ohe = housing_df[housing_df_nominal]
housing_df_ohe = pd.get_dummies(housing_df_ohe, drop_first=True)
## 168 Variables
housing_df_ohe_Temp = housing_df_ohe.sum().reset_index()
housing_df_ohe_Temp.columns = ['Name', 'response_Count']
housing_df_ohe_Temp = housing_df_ohe_Temp.loc[housing_df_ohe_Temp['response_Count']>10]
housing_df_ohe = housing_df_ohe[housing_df_ohe_Temp['Name']]

## removing the non numeric nominal variables which have now been one hot encoded
To_Be_removed = np.append(To_Be_removed, housing_df_nominal)
housing_df_linearReg = housing_df.drop(To_Be_removed, axis=1)
housing_df_linearReg = housing_df_linearReg.drop(['SalePrice'], axis=1)
housing_df_linearReg = pd.concat([housing_df_linearReg, housing_df_ohe],axis=1)
housing_df_linearReg['SalePrice'] = housing_df['SalePrice']

## Now we'll apply inrear regression ##
ind_var = housing_df_linearReg.iloc[:, 0:(housing_df_linearReg.shape[1]-2)]
dep_var = housing_df_linearReg.iloc[:, (housing_df_linearReg.shape[1]-1)]

## Linear Regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
x_train, x_test, y_train, y_test = train_test_split(ind_var, dep_var, test_size = .40, random_state=42)
reg = LinearRegression(normalize=True)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
tst = pd.DataFrame(reg.coef_, ind_var.columns)
## Cross Validation ##
cv = KFold(shuffle=True,n_splits=5)
scores = (np.sqrt(-cross_val_score(reg,ind_var,dep_var, cv = cv,scoring = 'neg_mean_squared_error')))
scores

### Looking at the scores outlier seem to be acting up. We'll try and remove the outliers which should hopefully fix the issues. 
### But Before that lets try using Ridge and Lasso Regression which are quite immune to outliers due to regulaization. 

def rmse_cv(model):
    rmse= np.sqrt(cross_val_score(model, ind_var, dep_var, scoring="explained_variance", cv = 5))
    # rmse= np.sqrt(-cross_val_score(model, ind_var, dep_var, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

alphas = [0.05, 0.1, 0.3, 1, 3, 6, 5, 7, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

ridgevis = pd.Series(cv_ridge, index=alphas) 
ridgevis = ridgevis.reset_index()
ridgevis.columns = ['Aphas', 'RMSE']
# ridgevis = ridgevis.loc[ridgevis['RMSE'] == np.min(ridgevis['RMSE'])]
ridgevis = ridgevis.loc[ridgevis['RMSE'] == np.max(ridgevis['RMSE'])]

ridge_model = Ridge(alpha = 6)
ridge_model.fit(x_train, y_train)
y_pred = ridge_model.predict(x_test.head(1))

## Creating a pickle file ##

pickle.dump(ridge_model, open('Ridge_model.pkl','wb'))
Ridge_model = pickle.load(open('Ridge_model.pkl', 'rb'))

pickle.dump(reg, open('Linear_reg_model.pkl','wb'))
Linear_reg_model = pickle.load(open('Linear_reg_model.pkl', 'rb'))

listofcol = x_test.dtypes.reset_index()
# res = (Ridge_model.predict([[6,3,3,3,3,0,0,2,1,1.5,3,8.06965530688616,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1]]))

res = (Linear_reg_model.predict([[6,3,3,3,3,0,0,2,1,1.5,3,8.06965530688616,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1]]))
print(math.exp(res))

data = housing_df[['BldgType', 'SalePrice']]
f, ax = plt.subplots(figsize=(7, 7))
fig = sns.boxplot(x='BldgType', y="SalePrice", data=data)


# print(np.array(x_test.iloc[:1,:]))


# resp_value = 12
# def response_to_numeric(resp_arr):
#     response_list = []
#     for resp in resp_arr:
#         if (resp_value == resp):
#             response_list = np.append(response_list, "1")
#         else:
#             response_list = np.append(response_list, "0")
#     return response_list.astype(int)
# response_to_numeric([1,2,3,4,5,6,7,8,9,10,11,12,13])

############
# housing_df.drop(To_Be_removed, axis=1).select_dtypes(exclude = 'object').columns
#########

# def chartoint(word):
#     wrd_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11} 
#     return(wrd_dict[word])

# testdf["experience"] = testdf["experience"].apply(lambda x: chartoint(x))
# print(testdf.dtypes)

# x = testdf.iloc[:,:-1]
# y = testdf.iloc[:,-1]

# linreg = LinearRegression()
# linreg.fit(x,y)

# pickle.dump(linreg, open("model.pkl","wb"))

# model = pickle.load(open("model.pkl", "rb"))
# print(model.predict([[3,4,5]]))



# X_Train, X_Test, y_Train, y_test = train_test_split(x,y,test_size = .30, random_state=52)
# regressor = LinearRegression()
# regressor.fit(X_Train, y_Train)
# y_pred = regressor.predict(X_Test)
# pickle.dump(regressor, open('model.pkl','wb'))

# model = pickle.load(open('model.pkl', 'rb'))
