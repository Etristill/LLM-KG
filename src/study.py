from participant import run_experiment, ParticipantInfo
import numpy as np
import pandas as pd
import random



def run_study(number_of_participants, participants_data_ranges: dict, trial_sequence, experiment_info):
    '''
    samples participants with characteristics within defined ranges
    ------------------------
    Params:
    number_of_participants (int): number of participants
    participants_data_ranges (dict): 'age': list(lowest age, highest age) and 'gender': list of genders
    trial_sequence (tuple): 
    experiment_info (ExperimentInfo): 

    Returns:
    DataFrame

    '''
    df = pd.DataFrame()
    for participant_id in range(number_of_participants):
        #create random participant
        participant_info = ParticipantInfo
        participant_info.age = random.choice(participants_data_ranges['age'])
        participant_info.gender = random.choice(participants_data_ranges['gender'])
        participant_info.id = participant_id

        #run experiment on participant
        participant_df = run_experiment(trial_sequence, participant_info, experiment_info)
    
        #update the df
        df = pd.concat(df, participant_df, axis = 0, ignore_index=False)
    return df


#examplar study

from experiment import generate_bandit_trials, ExperimentInfo

# specify experiment parameters
experiment_info = ExperimentInfo()
experiment_info.id = "1"
experiment_info.n_trials = 10
experiment_info.p_init = (0.7, 0.3)
experiment_info.sigma = (0.02, 0.02)
experiment_info.hazard_rate = 0.05

# specify participants' parameters

participants_data_ranges = {'age': [3, 50], 'gender': ['male', 'female']}


# Generate an experiment
trial_sequence = generate_bandit_trials(experiment_info=experiment_info)

# Run the study on the synthetic random sample
df = run_study(5, participants_data_ranges, trial_sequence, experiment_info)

df.to_csv("src/data/" + "experiment_" + experiment_info.id + ".csv", index=False)
