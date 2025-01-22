from sweetbean import Block, Experiment
from sweetbean.stimulus import Bandit, Text
from sweetbean.variable import (
    DataVariable,
    FunctionVariable,
    SharedVariable,
    SideEffect,
    TimelineVariable,
)
import random
import numpy as np

def generate_experiment(reward_sequence, generate_html=False):
    '''
    Generate a bandit task experiment with a given reward sequence.

    Parameters:
    reward_sequence (list): A numpy array of shape (n_trials, 2) where each row is a pair of rewards for the two bandits.
    '''

    # convert the reward sequence to a SweetBean timeline
    timeline = []
    for i in range(len(reward_sequence)):
        timeline.append(
            {
            "bandit_1": {"color": "orange", "value": int(reward_sequence[i][0])},
            "bandit_2": {"color": "blue", "value": int(reward_sequence[i][1])},
            }
        )

    # define instructions
    instruction = Text(
        text="In this experiment, you will see two boxes. Your goal is to find out which box gives more rewards by clicking on them. Each box might change how often it gives rewards, so keep experimenting and try to maximize your total rewards. When you select your response, answer only with '1' or '2' indicating the number of the bandit", duration=3000)

    # define bandit variables and stimuli
    bandit_1 = TimelineVariable("bandit_1")
    bandit_2 = TimelineVariable("bandit_2")

    score = SharedVariable("score", 0)
    value = DataVariable("value", 0)

    update_score = FunctionVariable(
        "update_score", lambda sc, val: sc + val, [score, value]
    )

    update_score_side_effect = SideEffect(score, update_score)

    bandit_task = Bandit(
        bandits=[bandit_1, bandit_2],
        side_effects=[update_score_side_effect],
    )

    # experiment blocks
    instruction_block = Block([instruction])
    trial_sequence = Block([bandit_task], timeline=timeline)
    experiment = Experiment([instruction_block, trial_sequence])
    if generate_html:
        experiment.to_html("bandit.html", path_local_download="bandit.json")
    return experiment


def generate_bandit_trials(n_trials: int, reward_prob=(0.5, 0.5)) -> np.ndarray:
    """
    Generates a random trial sequence for a two-armed bandit experiment.

    Args:
        n_trials (int): Number of trials.
        reward_prob (tuple): Probabilities of reward for each arm (arm_0, arm_1).

    Returns:
        np.ndarray: A (n_trials, 2) array where:
                    - Column 0: Chosen arm (0 or 1)
                    - Column 1: Reward received (0 or 1)
    """
    # Randomly select arms (0 or 1) for each trial
    chosen_arms = np.random.randint(0, 2, size=n_trials)

    # Generate rewards based on chosen arm and its probability
    rewards = np.array([
        np.random.rand() < reward_prob[arm] for arm in chosen_arms
    ], dtype=int)

    # Stack chosen arms and rewards into an (n_trials, 2) array
    trial_data = np.column_stack((chosen_arms, rewards))

    return trial_data

# Example usage:
n_trials = 10
reward_probabilities = (0.7, 0.3)  # Arm 0 has 70% reward probability, Arm 1 has 30%
trial_sequence = generate_bandit_trials(n_trials, reward_probabilities)

# Generate an experiment with the trial sequence
generate_experiment(trial_sequence)