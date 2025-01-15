#Linear Regression for predicting wickets
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Reading the IPL dataset
ipl_df = pd.read_csv('most_wickets_2022.csv')

# Splitting the dataset into features and target
X = ipl_df.iloc[:,1:10]
y = ipl_df.iloc[:, 10]


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the decision tree classifier model
lr = LinearRegression()

# Training the model using the training data
lr.fit(X_train, y_train)

# Predicting the target values for the test data
y_pred = lr.predict(X_test)
actual = y_test

new_player_data=pd.DataFrame({'Mat':[15],
                             'Inns':[15],
                             'Ov':[60],
                             'Runs':[440],
                             'Avg':[17.72],
                             'Econ':[8.00],
                             'SR':[12.42],
                             '4w':[2],
                             '5w':[1]})

predicted_performance = lr.predict(new_player_data)
print('predicted wickets taken: ',predicted_performance)

# Evaluating the model using various metrics
print("Mean Absolute Error:", mean_absolute_error(actual, y_pred))
print("Mean Squared Error:", mean_squared_error(actual, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(actual, y_pred)))
print("R-squared:", r2_score(actual, y_pred))

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
