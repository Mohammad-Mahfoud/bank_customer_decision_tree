import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score,accuracy_score
import matplotlib.pyplot as plt 




df = pd.read_csv("Dataset/Churn_Modelling.csv")


# dropping unneccesary columns : 
df = df.drop(["RowNumber","CustomerId","Surname"],axis=1)

# applying one hot encoding on str columns : 
df = pd.get_dummies(df,columns=["Geography","Gender"],dtype=int)


## splitting dataset into features and target: 
X = df.drop(["Exited"],axis=1)
y = df["Exited"]

# preparing the training and testing variables on features and target :
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


## initializing the tree model 

model = DecisionTreeClassifier(max_depth=5,random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# applying metrics analysis : 
f1score = f1_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print(f"F1 Score is {f1score}")
print(f"Accuracy Score is {accuracy}")


## visualizing the tree :
plt.figure(figsize=[20,8])

_ = plot_tree(model,feature_names=X.columns,filled=True,rounded=False,proportion=True,fontsize=11)
plt.show()