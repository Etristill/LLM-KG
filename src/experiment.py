#this is experiment.py

#he new implementation incorporates a weighted decay mechanism. 
# Probabilities drift toward a "center" (decay_center) at a rate determined by decay_rate.


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

class ExperimentInfo:
    """Structured information about 2-armed bandit experiments"""
    id: str
    n_trials: int
    p_init: Tuple[float, float]  # Initial reward probabilities for each arm
    sigma: Tuple[float, float]   # Standard deviation of random walk for each arm
    hazard_rate: float          # Probability of sudden changes in reward probabilities

def generate_experiment(experiment_info, generate_html=False):
    '''
    Generate a bandit task experiment with a given reward sequence.

    Parameters:
    reward_sequence (list): A numpy array of shape (n_trials, 2) where each row is a pair of rewards for the two bandits.
    '''
    reward_sequence = generate_bandit_trials(experiment_info=experiment_info)

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

def generate_bandit_trials(
    n_trials=100,
    p_init=(0.7, 0.3),
    sigma=(0.02, 0.02),
    hazard_rate=0.05,
    bounds=(0, 1),
    min_stable_trials=5,
    experiment_info=None
):
    """
    Generate a series of reward probabilities and sampled rewards for a 2-armed bandit with drifting probabilities and sudden changes.

    Parameters:
        n_trials (int): Number of trials.
        p_init (tuple): Initial reward probabilities for both arms (p1, p2).
        sigma (tuple): Standard deviation of the Gaussian noise added per trial for both arms (sigma1, sigma2).
        hazard_rate (float): Probability per trial of switching to completely new random reward probabilities.
        bounds (tuple): Lower and upper bounds for reward probabilities (default: (0, 1)).
        min_stable_trials (int): Minimum number of trials between change points.
        experiment_info (ExperimentInfo): Optional experiment info object containing parameters.

    Returns:
        np.ndarray: An (n_trials, 2) array of sampled rewards (0 or 1) for each arm.
    """
    # define function for updating the probability of a given arm according to a random walk process. 
    def random_walk_update(prob, sigma, bounds):
        # Add gaussian noise if sigma > 0
        noise = np.random.normal(0, sigma) if sigma > 1e-6 else 0
        new_prob = prob + noise
        return np.clip(new_prob, *bounds)
    
    # define function for pre-determining the change points of a given arm
    def sample_change_points(n_trials, hazard_rate, min_stable_trials):
        # Change points randomly sampled with hazard_rate probability
        change_points = []
        current_trial = min_stable_trials  # start after the minimum stable period
        while current_trial < n_trials:
            if np.random.random() < hazard_rate:
                change_points.append(current_trial)
            current_trial += 1
        return np.array(change_points)

    # get experiment settings / parameters if provided
    if experiment_info is not None:
        n_trials = experiment_info.n_trials
        p_init = experiment_info.p_init
        sigma = experiment_info.sigma
        hazard_rate = experiment_info.hazard_rate

    # declare local variables / initialise
    probabilities = np.zeros((n_trials, 2))
    rewards = np.zeros((n_trials, 2), dtype=int)
    probabilities[0] = p_init

    # sample change points for each arm
    change_points_1 = sample_change_points(n_trials, hazard_rate, min_stable_trials)
    change_points_2 = sample_change_points(n_trials, hazard_rate, min_stable_trials)
    change_points = [change_points_1, change_points_2]

    # iterate through trials
    for t in range(1, n_trials):
        # generate trial-wise reward probabilities for each arm
        for arm in range(2):
            if t in change_points[arm]:
                # Sudden change in probability
                probabilities[t, arm] = np.random.uniform(*bounds)
            else:
                # Gradual random walk
                probabilities[t, arm] = random_walk_update(
                    probabilities[t - 1, arm],
                    sigma[arm],
                    bounds
                )
        
        # sample rewards based on probabilities
        rewards[t] = [
            np.random.rand() < probabilities[t, 0],
            np.random.rand() < probabilities[t, 1]
        ]

    return rewards


if __name__ == "__main__":
    # Example usage:
    # specify experiment parameters
    experiment_info = ExperimentInfo()
    experiment_info.id = "test"
    experiment_info.n_trials = 100
    experiment_info.p_init = (0.7, 0.3)  # Arm 0 starts better
    experiment_info.sigma = (0.02, 0.02)  # Moderate drift
    experiment_info.hazard_rate = 0.05    # 5% chance of sudden changes

    # Generate trial sequence
    trial_sequence = generate_bandit_trials(experiment_info=experiment_info)

    # Alternatively, generate the entire experiment
    # generate_experiment(experiment_info)