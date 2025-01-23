#this is experiment.py


from typing import Dict, List, Optional, Set, Tuple, Any
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
from scipy.stats import beta


def merge_dicts(defaults, overrides):
    """Merges two dictionaries, giving precedence to overrides."""
    if overrides is None:
        overrides = {}
    return {**defaults, **overrides}

class ExperimentInfo:
    """Structured information about 2-armed bandit experiments"""
    id: int
    n_trials: int
    p_init: Tuple[float, float]
    sigma: Tuple[float, float]
    hazard_rate: float

def generate_experiment(experiment_info, generate_html=False):
    '''
    Generate a bandit task experiment with a given reward sequence.

    Parameters:
    reward_sequence (list): A numpy array of shape (n_trials, 2) where each row is a pair of rewards for the two bandits.
    '''
    reward_sequence = generate_bandit_trials(experiment_info)

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


def generate_bandit_trials(n_trials=100, p_init=(0.5, 0.5), sigma=(0.02, 0.02), hazard_rate=0.05, bounds=(0, 1), experiment_info=None):
    """
    Generate a series of reward probabilities and sampled rewards for a 2-armed bandit with drifting probabilities and sudden changes.

    Parameters:
        n_trials (int): Number of trials.
        p_init (tuple): Initial reward probabilities for both arms (p1, p2).
        sigma (tuple): Standard deviation of the Gaussian noise added per trial for both arms (sigma1, sigma2).
        hazard_rate (float): Probability per trial of switching to completely new random reward probabilities.
        bounds (tuple): Lower and upper bounds for reward probabilities (default: (0, 1)).

    Returns:
        tuple:
            np.ndarray: An (n_trials, 2) array of reward probabilities for each arm over time.
            np.ndarray: An (n_trials, 2) array of sampled rewards (0 or 1) for each arm.
    """
    if experiment_info is not None:
        n_trials = experiment_info.n_trials
        p_init = experiment_info.p_init
        sigma = experiment_info.sigma
        hazard_rate = experiment_info.hazard_rate

    p1, p2 = p_init
    sigma1, sigma2 = sigma

    probabilities = np.zeros((n_trials, 2))
    rewards = np.zeros((n_trials, 2), dtype=int)
    probabilities[0] = [p1, p2]

    for t in range(1, n_trials):
        if np.random.rand() < hazard_rate:
            # Change to completely new random probabilities
            p1, p2 = np.random.uniform(bounds[0], bounds[1], 2)
        else:
            # Drift normally
            p1 = np.clip(p1 + np.random.normal(0, sigma1), bounds[0], bounds[1])
            p2 = np.clip(p2 + np.random.normal(0, sigma2), bounds[0], bounds[1])

        probabilities[t] = [p1, p2]
        rewards[t] = [np.random.rand() < p1, np.random.rand() < p2]


    return rewards


# Example usage:

# specify experiment parameters
experiment_info = ExperimentInfo()
experiment_info.id = "1"
experiment_info.n_trials = 10
experiment_info.p_init = (0.7, 0.3)
experiment_info.sigma = (0.02, 0.02)
experiment_info.hazard_rate = 0.05

trial_sequence = generate_bandit_trials(experiment_info=experiment_info)

# Alternatively, we can generate the entire experiment
# generate_experiment(experiment_info)