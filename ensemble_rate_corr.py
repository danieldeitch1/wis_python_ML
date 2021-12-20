import numpy as np

def ensemble_rate_corr(data,trial_id):
    """
    Calculated the averaged neuronal activity for each neuron in each trial 
    and removing trials without any neuronal activity
    """
    # define an empty array that will contain the averaged neuronal activity in each trial
    ensemble_rate = np.zeros([len(data),int(max(trial_id))])
    
    # define an array with trial IDs 
    trial_list = np.arange(int(min(trial_id)),int(max(trial_id))+1)
    
    # average the neuronal activity in each trial to create a matrix in the size of n by t (number of neurons by number of trials)
    for trial in np.arange(0,len(trial_list)): # loop over trials
        current_trial = trial_id == trial_list[trial] 
        current_trial_activity = data[:,current_trial] # subset activity of a single trial
        if current_trial_activity.size != 0:
            ensemble_rate[:,trial] = current_trial_activity.mean(1) # average the neuronal activity over time for each neuron
        
    # remove trials with no activity  
    valid_trials = ensemble_rate.mean(0) > 0 
    valid_ensemble_rate = ensemble_rate[:,valid_trials]
    return valid_ensemble_rate
