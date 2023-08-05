# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# spliting data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NAN,strategy="mean",)
X[ : ,1:3] = imputer.fit_transform(X[ : ,1:3]) 


# convert data from string to int or binary 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer
ctX = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough')
X = ctX.fit_transform(X)
lableencoder_y = LabelEncoder()
y = lableencoder_y.fit_transform(y)


#spliting dataset to training data and testing data
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# feature scaling  => tow way standerd deviation or normalization

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

