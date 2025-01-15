#SVR for predicting runs(sensitivity, specificity)
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


ipl_df = pd.read_csv('most_runs_2022.csv')

X = ipl_df.iloc[:,1:12]
y = ipl_df.iloc[:, 12]

# Encode categorical features
le = LabelEncoder()
X['HS'] = le.fit_transform(X['HS'])
X['Avg'] = le.fit_transform(X['Avg'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr = SVR(kernel='linear')

svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)
actual = y_test

new_player_data=pd.DataFrame({'Mat':[18],
                              'Inns':[18],
                              'NO':[5],
                              'HS':[149],
                              'Avg':[50.23],
                              'BF':[581],
                              'SR':[145.18],
                              '100':[5],
                              '50':[6],
                              '4s':[82],
                              '6s':[10]})

predicted_performance = svr.predict(new_player_data)

print('predicted runs scored : ',predicted_performance)

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


