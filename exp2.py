import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import os
print ("working directory ", os.getcwd())
print ("files ", os.listdir())
balance_data = pd.read_csv("decision_tree_dataset.csv")
print ("dataset length", len(balance_data))
print ("dataset shape :", balance_data.shape)
print(balance_data.head())
balance_data.iloc[:,-1]=balance_data.iloc[:,-1].astype(str).str.lower()
X= balance_data.iloc[:,:-1]
Y= balance_data.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=100)
model = DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_leaf=5, random_state=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print ("accuracy (%): ", accuracy_score(Y_test,Y_pred)*100)
print (confusion_matrix(Y_test, Y_pred))
sample1 = pd.DataFrame([[3815, 1407, 594, 4]]) 
sample2 = pd.DataFrame([[9000,3500,780,13]])
print ("prediction 1: ", model.predict(sample1))
print ("prediction 2: ", model.predict(sample2))

# ---- Decision Tree Visualization ----

tree.plot_tree(model,feature_names=X.columns, class_names=model.classes_, filled=True) 

plt.show()
