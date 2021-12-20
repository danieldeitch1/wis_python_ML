# the analysed data is publicly available and was downlaoded from:
# 'Diamanti, E. M., Reddy, C. B., Schröder, S., Muzzu, T., Harris, K. D., Saleem, A. B., & Carandini, M. (2021). 
# Spatial modulation of visual responses arises in cortex with active navigation. Elife, 10, e63705.‏'

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
from ensemble_rate_corr import ensemble_rate_corr
import numpy as np

# define path for directory containing both 'neuronal_data.npy' and 'trial_info.npy'
data_path = os.path.dirname(__file__)

# load neuronal data of a single mouse recorded from visual area V1
data = np.load(os.path.join(data_path, 'neuronal_data.npy'))

# load trial information
trial_id = np.load(os.path.join(data_path, 'trial_info.npy'))
trial_id = trial_id[:,0]

# calculate the averaged activity rate for each neuron in each trial
valid_ensemble_rate = ensemble_rate_corr(data,trial_id)

# plot the relationship between neuronal activity in trial 2 and neuronal activity in trial 3
# each dot is a neuron and the values on each axis is the neuronal activity in that trial
plt.figure()
plt.scatter(valid_ensemble_rate[:,1],valid_ensemble_rate[:,2])
plt.xlabel("Neuronal activity in trial 2")
plt.ylabel("Neuronal activity in trial 3")

# fit a linear model using the neuronal activity in trial 2 to predict the neuronal activity in trial 3
x_data = valid_ensemble_rate[:,1] # neuronal activity of trial 2
y_data = valid_ensemble_rate[:,2] # neuronal activity of trial 3
model = LinearRegression()
model.fit(x_data.reshape(-1, 1),y_data)
print('coefficient of determination: ',model.score(x_data.reshape(-1, 1),y_data)) # R2 of 0.585
# The model revealed that approximately 59% of the variance in the neuronal activity in tria 3
# can be explained by using the neuronal activity in trial 2

# test the the repreducability of the model by a split-test approach
# 80% of the data will be used for training the model and the rest 20% to testing it.
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)
print('coefficient of determination: ',model.score(X_train.reshape(-1, 1), y_train)) # R2 of 0.576
# The analysis revealed comaprible explained variance when using 80% of the data compared to when using the entire dataset

# predict the neuronal activity of the test data
y_pred = model.predict(X_test.reshape(-1, 1))

# scatter plot demonstrating the linear relationship between predicted neuronal 
# activity by the model and the actual measured neuronal activiy
plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel("Actual (measured) neuronal activity")
plt.ylabel("Predicted neuronal activity")

# fit linear model between real and predicted neuronal activity
model = LinearRegression()
model.fit(y_test.reshape(-1, 1), y_pred)
print('coefficient of determination: ',model.score(y_test.reshape(-1, 1), y_pred)) # R2 of 0.644
# There is a strong correlation (R2=0.64) between the neuronal activity predicted by the model
# and the neuronal activity measured by the experimenters

# fit multiple regression model to predict the neuronal activity in trial 4 
# based on the neuronal activity of both trial 2 and trial 3:
x_data = valid_ensemble_rate[:,1:3]
y_data = valid_ensemble_rate[:,3]
model = LinearRegression()
model.fit(x_data, y_data)
print('coefficient of determination: ',model.score(x_data, y_data)) # R2 of 0.21
# It seems that using the activity of both trials 2 and trial 3 explain only 20%
# of the variance in the neuronal activity of trial 4.
