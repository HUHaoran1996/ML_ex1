import pandas as pd
import numpy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


os.chdir('C:\\Users\\dell\\Desktop')
data=pd.read_csv('2008.csv')
print (data.shape)
print (data['ArrDelay'].isnull().sum())
data['ArrDelay'].fillna(-1, inplace=True)


# #Delete unwanted data columns

data=data.drop(['UniqueCarrier','Origin','TailNum','Dest','CancellationCode'],axis=1)
data.isnull().sum()

data=data.fillna(0)
ArrDelay=data.pop('ArrDelay')
CarrierDelay=data.pop('CarrierDelay')
WeatherDelay=data.pop('WeatherDelay')
NasDelay=data.pop('NASDelay')
SecurityDelay=data.pop('SecurityDelay')
LateAircraftDelay=data.pop('LateAircraftDelay')
AllDelay=ArrDelay+CarrierDelay+WeatherDelay+NasDelay+SecurityDelay+LateAircraftDelay

# # Scaling and Normalzing Data

# # Standard Scaling

scaler=StandardScaler().fit(data)
std_scaled = scaler.transform(data)

# # Min-Max Scaling

minmax_scaler=MinMaxScaler(feature_range=(0,1))
minmax_scaled = minmax_scaler.fit_transform(data)

# #Machine Learning (Standard Scaled Data Train/Test Split)
# #ArrDelay

print ("Predicting Arr Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, ArrDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)



# #CarrierDelay

print ("Predicting Carrier Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, CarrierDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)


# #WeatherDelay

print ("Predicting Weather Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, WeatherDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #NasDelay

print ("Predicting Nas Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, NasDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #Security Delay

print ("Predicting Security Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, SecurityDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #LateAircraftDelay

print ("Predicting Late Aircraft Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, LateAircraftDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #All Delay

print ("Predicting AllDelay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(std_scaled, AllDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #Machine Learning (Minmax Scaled Data Train/Test Split)
# #ArrDelay

print ("Predicting Arr Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled, ArrDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)



# #CarrierDelay

print ("Predicting Carrier Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled , CarrierDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)


# #WeatherDelay

print ("Predicting Weather Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled , WeatherDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #NasDelay

print ("Predicting Nas Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled , NasDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #Security Delay

print ("Predicting Security Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled , SecurityDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #LateAircraftDelay

print ("Predicting Late Aircraft Delay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled, LateAircraftDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)

# #All Delay

print ("Predicting AllDelay")
# # Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(minmax_scaled, AllDelay, test_size=0.2, random_state=42)

# # Linear regression

print ("Using Linear Regression")
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
lin_mse=mean_squared_error(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", lin_mse)
lin_rmse = numpy.sqrt(lin_mse)
print(lin_rmse)
# # Decision tree

print ("Using Decision Trees Regression")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
predictions = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test,predictions)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
print ("Error on Testing is : ", tree_mse)
tree_rmse = numpy.sqrt(tree_mse)
print(tree_rmse)