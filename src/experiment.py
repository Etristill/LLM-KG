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

class ExperimentInfo:
    """Structured information about 2-armed bandit experiments"""
    id: int
    n_trials: int
    p_init: Tuple[float, float]
    sigma: Tuple[float, float]
    decay_rate: Tuple[float, float]
    decay_center: Tuple[float, float]
    hazard_rate: Tuple[float, float]

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


def generate_bandit_trials(
    n_trials=100,
    p_init=(0.5, 0.5),
    sigma=(0.02, 0.02),
    decay_rate=(0.05, 0.05),
    decay_center=(0.5, 0.5),
    hazard_rate=(0.05, 0.05),
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
        decay_rate (tuple): Rate at which the reward probabilities decay towards some "default" values (decay_center) for both arms (decay_rate1, decay_rate2).
        decay_center (tuple): The "default" values which the reward probabilities decay towards, according to the decay_rate, for both arms (decay_center, decay_center).
        hazard_rate (float): Probability per trial of switching to completely new random reward probabilities for both arms (hazard_rate1, hazard_rate2).
        bounds (tuple): Lower and upper bounds for reward probabilities (default: (0, 1)).

    Returns:
        tuple:
            np.ndarray: An (n_trials, 2) array of reward probabilities for each arm over time.
            np.ndarray: An (n_trials, 2) array of sampled rewards (0 or 1) for each arm.
    """
    # define function for updating the probability of a given arm according to a random walk process
    def random_walk_update(prob, sigma, decay_rate, decay_center, bounds):
        # only add zero-mean Gaussian noise if standard deviation sigma is greater than zero
        if sigma > 1e-6:
            noise = np.random.normal(0, sigma)
        else:
            noise = 0
        # Daw et al. (2006), Nature. Updated probability is weighted sum of current probability
        # and "decay center", plus noise.
        # NB decay rate of zero corresponds to standard random walk without decay.        
        new_prob = (1 - decay_rate) * prob + decay_rate * decay_center + noise
        return np.clip(new_prob, *bounds)
    
    # define function for pre-determining the change points of a given arm
    def sample_change_points(n_trials, hazard_rate, min_stable_trials):
        # Nassar et al. (2010), Journal of Neuroscience. Change points randomly sampled from an
        # exponential distribution with rate of 0.05, corresponding to a mean of 1/0.05 = 20 trials.
        change_points = []
        current_trial = min_stable_trials  # start after the minimum stable period
        while current_trial < n_trials:
            # NB the numpy implementation of the exponential distribution is parameterised in terms
            # of the "scale", which is the _inverse_ rate, which is equal to the mean.
            interval = np.random.exponential(1 / hazard_rate)
            current_trial += int(np.ceil(interval))
            if current_trial < n_trials:
                change_points.append(current_trial)
        return np.array(change_points)


    # get experiment settings / parameters
    if experiment_info is not None:
        n_trials = experiment_info.n_trials
        p_init = experiment_info.p_init
        sigma = experiment_info.sigma
        decay_rate = experiment_info.decay_rate
        decay_center = experiment_info.decay_center
        hazard_rate = experiment_info.hazard_rate

    # declare local variables / initialise
    probabilities = np.zeros((n_trials, 2))
    rewards = np.zeros((n_trials, 2), dtype=int)
    probabilities[0] = p_init

    # sample change points for each arm, skipping if hazard rate is zero
    change_points = [
        sample_change_points(n_trials, hazard_rate[i], min_stable_trials)
        for i in range(2) if hazard_rate[i] > 1e-6
    ]

    # iterate through trials
    for t in range(1, n_trials):
        # generate trial-wise reward probabilities for each arm
        for arm in range(2):
            if t in change_points[arm]:
                probabilities[t, arm] = np.random.uniform(*bounds)
            else:
                probabilities[t, arm] = random_walk_update(
                    probabilities[t - 1, arm],
                    sigma[arm],
                    decay_rate[arm],
                    decay_center[arm],
                    bounds
                )
        # sample rewards based on probabilities
        rewards[t] = [
            np.random.rand() < probabilities[t, 0],
            np.random.rand() < probabilities[t, 1]
        ]

    # TODO do we also want the option to return the probabilities?
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