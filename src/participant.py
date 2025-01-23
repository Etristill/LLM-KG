import openai
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from experiment import ExperimentInfo

class ParticipantInfo:
    """Structured information about 2-armed bandit experiments"""
    id: int
    age: int
    gender: str

def generate(prompt):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    message = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response = message.choices[0].message.content

    return response[0]

def execute_experiment(experiment):
    '''
    Run the experiment on the synthetic participant.
    '''
    data = experiment.run_on_language(generate)
    print(data)
    return data

def run_experiment(trial_sequence, participant_info, experiment_info):

    age = participant_info.age
    gender = participant_info.gender
    participant_id = participant_info.id

    n_trials = len(trial_sequence)
    rewards = np.zeros(n_trials)
    choices = np.zeros(trial_sequence.shape)

    prompt = ("You are a human participant with the following demographics: "
              + str(age) + " years old and " + str(gender) + "."
              + " You are participating in the following psychology experiment. ")
    prompt += "In this experiment, you will see two boxes. Your goal is to find out which box gives more rewards by clicking on them. Each box might change how often it gives rewards, so keep experimenting and try to maximize your total rewards. When you select your response, answer only with '1' or '2' indicating the number of the bandit. "

    for trial in range(n_trials):
        prompt += "You see 2 bandits, Bandit 1 and Bandit 2. Choose a bandit by naming the number of the bandit. "
        response = generate(prompt + "You name ")
        llm_choice = int(response)
        transformed_choice = llm_choice - 1
        reward = int(trial_sequence[trial][transformed_choice])
        choices[trial][transformed_choice] = 1
        rewards[trial] = reward
        prompt += "You chose bandit " + str(llm_choice) + ". You received a reward of " + str(reward) + ". "

    choices = np.where(choices == 1)[1]
    df = pd.DataFrame({"trial": np.arange(n_trials), "choice": choices, "reward": rewards,
                       "session": participant_id, "age": age, "gender": gender,
                       "experiment_id": experiment_info.id, "n_trials": n_trials,
                       "p_init": str(experiment_info.p_init), "sigma": str(experiment_info.sigma),
                       "hazard_rate": str(experiment_info.hazard_rate)})


    return df


# Example usage
from experiment import generate_experiment, generate_bandit_trials

# specify experiment parameters
experiment_info = ExperimentInfo()
experiment_info.id = "1"
experiment_info.n_trials = 10
experiment_info.p_init = (0.7, 0.3)
experiment_info.sigma = (0.02, 0.02)
experiment_info.hazard_rate = 0.05

# specify participant parameters

participant_info = ParticipantInfo()
participant_info.id = "1"
participant_info.age = 34
participant_info.gender = "male"

# Generate an experiment
trial_sequence = generate_bandit_trials(experiment_info=experiment_info)

# Run the experiment on the synthetic participant
df = run_experiment(trial_sequence, participant_info, experiment_info)

df.to_csv("src/data/" + "experiment_" + experiment_info.id + ".csv", index=False)

# Alternative using SweetBean
# data = execute_experiment(experiment)
# print(data)

# session, choice, reward