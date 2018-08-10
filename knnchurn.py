# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:51:52 2018

@author: MANASI KULKARNI
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#from xgboost import XGBClassifier
#from sklearn.ensemble import 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib

import matplotlib.pyplot as plt
from IPython.display import display, HTML
df = pd.read_csv("churn-bigml-80.csv")
display(df.head(5))
print("Number of rows: ", df.shape[0])
counts = df.describe().iloc[0]
display(
    pd.DataFrame(
        counts.tolist(), 
        columns=["Count of values"], 
        index=counts.index.values
    ).transpose()
)
df = df.drop([ "State","International plan","Voice mail plan"], axis=1)
features = df.drop(["Churn"], axis=1).columns

from sklearn.cross_validation import train_test_split  
df_train, df_test = train_test_split(df, test_size=0.20)

from sklearn.naive_bayes import GaussianNB 
classifier =GaussianNB ()
classifier.fit(df_train[features], df_train["Churn"])  


# Make predictions
predictions = classifier.predict(df_test[features])
probs = classifier.predict_proba(df_test[features])
display(predictions)
score = classifier.score(df_test[features], df_test["Churn"])
print("Accuracy: ", score)
get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_test["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(confusion_matrix)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_test["Churn"], probs[:,1])
plt.title('Customer Churn Prediction NV ')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111)

df_f = pd.DataFrame(classifier.class_count_, columns=['importance'])
for ft in features:
    print(ft)
df_f["labels"] = ft 
df_f.sort_values('importance',inplace=True, ascending=False)
display(df_f.head(5))

index = np.arange(len(classifier.class_count_))
bar_width = 0.3
rects = plt.barh(index ,df_f['importance'], bar_width, alpha=0.7, color='rb', label='Main')
plt.yticks(index, df_f["labels"])
plt.show()
df_test["prob_true"] = probs[:, 1]
df_risky = df_test[df_test["prob_true"] > 0.9]
display(df_risky.head(5)[["prob_true"]])