#RNN WICKETS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 

#Reading the IPL dataset
ipl_df = pd.read_csv('most_wickets_2022.csv')

#Splitting the dataset into features and target
X = ipl_df.iloc[:,1:10]
y = ipl_df.iloc[:, 10]

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardizing the input data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Reshaping the input data for RNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#Defining the RNN model
rnn_model = keras.Sequential(
    [
        layers.LSTM(64, activation="relu", input_shape=(X_train.shape[1], 1)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="linear")
    ]
)

#Compiling the RNN model
rnn_model.compile(loss='mse', optimizer='adam', metrics=['mse'])

#Training the RNN model
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32)

#Predicting the target values for the test data
y_pred_rnn = rnn_model.predict(X_test)
y_pred_rnn = y_pred_rnn.reshape(y_pred_rnn.shape[0])

#Calculate and print the mean absolute error of the predictions
mae_rnn = sum(abs(y_test - y_pred_rnn))/len(y_test)
print("RNN Mean Absolute Error:", mae_rnn)

#Calculate and print the model's performance metrics
mse_rnn = keras.metrics.mean_squared_error(y_test, y_pred_rnn).numpy()
r2score_rnn = 1 - mse_rnn/np.var(y_test)
print("RNN Mean squared error:", mse_rnn)
print("RNN R2 score:", r2score_rnn)

#Calculate and print the model's sensitivity, specificity, and accuracy
cutoff = y_train.mean()
y_pred_binary = (y_pred_rnn >= cutoff).astype(int)
y_test_binary = (y_test >= cutoff).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(y_test_binary, y_pred_binary)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Accuracy:", accuracy)

# create a scatter plot of actual values
plt.scatter(y_test, y_test, color='blue', label='Actual')

# create a scatter plot of predicted values
plt.scatter(y_test, y_pred_rnn, color='green', label='Predicted')

# set the axis labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

# add a legend
plt.legend()

#add a diagonal line to the plot
axes=plt.gca()
x_values=np.array(axes.get_xlim())
y_values=x_values
plt.plot(x_values,y_values,'--',color='red')

#display the plot
plt.show()
