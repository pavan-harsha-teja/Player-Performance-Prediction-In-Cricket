#Random Forest for predicting wickets
#Importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error,r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import graphviz
import matplotlib.pyplot as plt

#Reading the IPL dataset
ipl_df = pd.read_csv('most_wickets_2022.csv')

#Splitting the dataset into features and target
X = ipl_df.iloc[:,1:10]
y = ipl_df.iloc[:, 10]



#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating the Random Forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

#Training the model using the training data
rf.fit(X_train, y_train)

#Predicting the target values for the test data
y_pred = rf.predict(X_test)
actual = y_test

#Calculate the mean absolute error of the predictions
mae = sum(abs(y_test - y_pred))/len(y_test)
print("Mean Absolute Error:", mae)

# create a scatter plot of actual values
plt.scatter(y_test, y_test, color='blue', label='Actual')

# create a scatter plot of predicted values
plt.scatter(y_test, y_pred, color='green', label='Predicted')

# set the axis labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

# add a legend
plt.legend()

# add a diagonal line to the plot
axes=plt.gca()
x_values=np.array(axes.get_xlim())
y_values=x_values
plt.plot(x_values,y_values,'--',color='red')

# display the plot
plt.show()


#Feature importances plot
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

#Print feature rankings
print("Feature rankings:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

#Plot feature importances
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

#Calculate and print the model's performance metrics
mse = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)
print("Mean squared error:", mse)
print("R2 score:", r2score)

#Calculate and print the model's sensitivity, specificity, and accuracy
cutoff = y_train.mean()
y_pred_binary = (y_pred >= cutoff).astype(int)
y_test_binary = (y_test >= cutoff).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(y_test_binary, y_pred_binary)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Accuracy:", accuracy)

#Print the confusion matrix and classification report

print("Confusion Matrix:")
print(confusion_matrix(y_test_binary, y_pred_binary))
print("Classification Report:")
print(classification_report(y_test_binary, y_pred_binary))
