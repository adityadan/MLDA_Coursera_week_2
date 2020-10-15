from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

os.chdir("C:/Users/Aditya/mlda")


AH_data = pd.read_csv("Cat_stats.csv")
data_clean = AH_data.dropna()

data_clean.dtypes
data_clean.describe()

predictors = data_clean[['Body_length', 'Tail_length', 'Height', 'Weight', 'Tail_texture', 'Coat_colour']]

targets = data_clean.Wildcat

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size = .4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 5)
classifier = classifier.fit(pred_train,tar_train)

predictions = classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
print(model.feature_importances_)

trees = range(5)
accuracy = np.zeros(5)

for idx in range(len(trees)):
   classifier = RandomForestClassifier(n_estimators = idx + 1)
   classifier = classifier.fit(pred_train, tar_train)
   predictions = classifier.predict(pred_test)
   accuracy[idx] = sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)