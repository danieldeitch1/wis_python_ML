import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# the analysed data is publicly available and was downlaoded from:
# 'Diamanti, E. M., Reddy, C. B., Schröder, S., Muzzu, T., Harris, K. D., Saleem, A. B., & Carandini, M. (2021). 
# Spatial modulation of visual responses arises in cortex with active navigation. Elife, 10, e63705.‏'

# define path for directory containing both 'neuronal_data.npy' and 'trial_info.npy'
data_path = r'C:\Users\rotaition\Desktop\wis_python'

# load neuronal data of a single mouse recorded from visual area V1
data = np.load(data_path+'\\neuronal_data.npy')

# load trial information
trial_id = np.load(data_path+'\\trial_info.npy')
trial_id = trial_id[:,0]

# define an empty array that will contain the averaged neuronal activity in each trial
ensemble_rate = np.zeros([len(data),int(max(trial_id))])

# define an array with trial IDs 
trial_list = np.arange(int(min(trial_id)),int(max(trial_id))+1)

# average the neuronal activity in each trial to create a matrix in the size of n by t (number of neurons by number of trials)
for trial in np.arange(0,len(trial_list)): # loop over trials
    current_trial = trial_id == trial_list[trial] 
    current_trial_activity = data[:,current_trial] # subset activity of a single trial
    ensemble_rate[:,trial] = current_trial_activity.mean(1) # average the neuronal activity over time for each neuron
    
# remove trials with no activity
valid_trials = ensemble_rate.mean(0) > 0 
valid_ensemble_rate = ensemble_rate[:,valid_trials] 

# plot the relationship between neuronal activity in trial 2 and neuronal activity in trial 3
# each dot is a neuron and the values on each axis is the neuronal activity in that trial
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
