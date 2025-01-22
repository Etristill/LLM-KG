import openai
from dotenv import load_dotenv
import os
import numpy as np

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

    return response

def execute_experiment(experiment):
    '''
    Run the experiment on the synthetic participant.
    '''
    data = experiment.run_on_language(generate)
    print(data)
    return data

def execute_experiment_simple(trial_sequence, age, gender):

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

    return rewards, choices


# Example usage
from experiment import generate_experiment, generate_bandit_trials

# specify experiment parameters
n_trials = 10
reward_probabilities = (0.7, 0.3)  # Arm 0 has 70% reward probability, Arm 1 has 30%
trial_sequence = generate_bandit_trials(n_trials, reward_probabilities)

# Generate an experiment with the trial sequence
experiment = generate_experiment(trial_sequence)

# Run the experiment on the synthetic participant
# data = execute_experiment(experiment)
rewards, choices = execute_experiment_simple(trial_sequence, "18", "male")

print(trial_sequence)
print(rewards)
print(choices)