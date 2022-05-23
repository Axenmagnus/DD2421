#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:20:53 2022

@author: magnusaxen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:00:33 2022

@author: magnusaxen
"""#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

train=pd.read_csv("TrainOnMe-4.csv")
train=train.drop_duplicates()
orgtrain=train
#train.isnull().sum()
train = train.dropna()
train.head()
train.pop("Unnamed: 0")
from sklearn.preprocessing import StandardScaler
features = list(train.columns)
features.remove("et")
features.remove("y")


x = train.loc[:, features].values
# Separating out the target
y = train.loc[:,['y']].values


finalDf=train


finalDf["A"] = finalDf.et.where(finalDf.et=="A", "0")

finalDf.loc[finalDf.A != 'A', 'A'] = 0
finalDf.loc[finalDf.A == 'A', 'A'] = 1

finalDf["B"] = finalDf.et.where(finalDf.et=="B", "0")
finalDf.loc[finalDf.B != 'B', 'B'] = 0
finalDf.loc[finalDf.B == 'B', 'B'] = 1
finalDf["I"] = finalDf.et.where(finalDf.et=="I", "0")
finalDf.loc[finalDf.I != 'I', 'I'] = 0
finalDf.loc[finalDf.I == 'I', 'I'] = 1
finalDf["W"] = finalDf.et.where(finalDf.et=="W", "0")
finalDf.loc[finalDf.W != 'W', 'W'] = 0
finalDf.loc[finalDf.W == 'W', 'W'] = 1
finalDf.pop("et")
finalDf.pop("y")

X=finalDf
finalDf = pd.concat([finalDf, orgtrain[['y']]], axis = 1)


random_state = 15
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size = 0.3,
                     shuffle = True,
                     random_state=random_state)
#---train the model using Logistic Regression---
log_reg = LogisticRegression(max_iter = 5000)
log_reg.fit(X_train, y_train)
#---evaluate the model---
pred=log_reg.score(X_test,y_test)
print(log_reg.score(X_test,y_test))
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


lr_list = [0.1]
N_list=[30]
for learning_rate in lr_list:
    for nest in N_list:
        gb_clf = GradientBoostingClassifier(n_estimators=nest, learning_rate=learning_rate, max_features=3, max_depth=10, random_state=12)
        gb_clf.fit(X_train, y_train)
    
        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


predictions = gb_clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))




#Ran previous calculations with various values to optimize

#Now lets evaluate

FinalTest=pd.read_csv("EvaluateOnMe-4.csv")
#Need adjust it for the model as before
FinalTest.pop("Unnamed: 0")
FinalTest["A"] = FinalTest.et.where(FinalTest.et=="A", "0")

FinalTest.loc[FinalTest.A != 'A', 'A'] = 0
FinalTest.loc[FinalTest.A == 'A', 'A'] = 1

FinalTest["B"] = FinalTest.et.where(FinalTest.et=="B", "0")
FinalTest.loc[FinalTest.B != 'B', 'B'] = 0
FinalTest.loc[FinalTest.B == 'B', 'B'] = 1
FinalTest["I"] = FinalTest.et.where(FinalTest.et=="I", "0")
FinalTest.loc[FinalTest.I != 'I', 'I'] = 0
FinalTest.loc[FinalTest.I == 'I', 'I'] = 1
FinalTest["W"] = FinalTest.et.where(FinalTest.et=="W", "0")
FinalTest.loc[FinalTest.W != 'W', 'W'] = 0
FinalTest.loc[FinalTest.W == 'W', 'W'] = 1
FinalTest.pop("et")
lastpredictions = gb_clf.predict(FinalTest)


np.savetxt("file1.txt", lastpredictions,fmt='%s')



