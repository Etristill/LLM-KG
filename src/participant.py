import openai
from dotenv import load_dotenv
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


import os


# Example usage
from experiment import generate_experiment, generate_bandit_trials

# specify experiment parameters
n_trials = 10
reward_probabilities = (0.7, 0.3)  # Arm 0 has 70% reward probability, Arm 1 has 30%
trial_sequence = generate_bandit_trials(n_trials, reward_probabilities)

# Generate an experiment with the trial sequence
experiment = generate_experiment(trial_sequence)

# Run the experiment on the synthetic participant
data = execute_experiment(experiment)