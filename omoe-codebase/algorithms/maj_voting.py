from math import prod  # Python 3.8+; for earlier versions, use functools.reduce(operator.mul,...)
import numpy as np
from config.variables import *
from itertools import product
import math
import math
from itertools import combinations

def compute_probability_products(y_list, p_list):
    """
    Compute the product of probabilities (or their complements) for each sublist in y_list.
    
    For each element in a sublist of y_list:
    - If the element is 1, multiply the corresponding probability from p_list.
    - If the element is 0, multiply the complement of the corresponding probability from p_list (1 - p).
    
    Args:
        y_list (list of lists): Binary sublists (e.g., [[0, 1, 0, 1, 0],...]).
        p_list (list): Probabilities (e.g., [0.4, 0.3, 0.2, 0.15, 0.11]).
    
    Returns:
        list: Products of probabilities (or their complements) for each sublist in y_list.
    """
    return [
        prod(
            p if y else (1 - p)  # Use probability if y=1, else use its complement
            for y, p in zip(sublist, p_list)  # Pair each y with corresponding p
        )
        for sublist in y_list  # Iterate over all sublists in y_list
    ]

def p_maj_vote(p, weights=None, quota=None, voting_epsilon=VOTING_EPSILON):
    total_probability = 0
    
    # Generate all possible outcomes (correct/incorrect for each expert)
    n = len(p)

    if weights is None:
        weights = [1] * n
        voting_epsilon = 0.0
    
    if quota is None:
        quota = np.sum(weights) / 2
    
    for outcome in product([0, 1], repeat=n):
        # Calculate the probability of this outcome
        prob = 1.0
        for i in range(n):
            prob *= p[i] if outcome[i] == 1 else (1 - p[i])
        
        # Calculate the sum of weights for correct predictions
        weight_sum = sum(weights[i] for i in range(n) if outcome[i] == 1)
        
        # Check if the sum of weights exceeds the quota

        if weight_sum > quota - voting_epsilon:
            total_probability += prob
    
    return total_probability

def find_opt_egal_voting(expert_p):
    sorted_p = sorted(expert_p, reverse=True)
    max_prob = -np.Inf
    best_sublist = []
    
    # Iterate over all possible top-K sublists
    for k in range(1, len(sorted_p) + 1):
        sublist = sorted_p[:k]
        # sublist_p = [e.p for e in sublist]
        current_prob = p_maj_vote(sublist, quota=len(sublist) / 2)
        
        # Update the best configuration if current_prob is higher
        if current_prob > max_prob:
            max_prob = current_prob
            best_sublist = sublist.copy()
    
    return best_sublist, max_prob

    
def empirical_is_majority_win(expert_p, weights=None, Q=None):
    """
    Simulates one empirical draw of expert votes and checks if the sum of correct votes > Q.
    
    Args:
        expert_p (list): List of probabilities that each expert votes correctly.
        Q (float): Quota (typically len(expert_p)/2 for majority).
    
    Returns:
        bool: True if the sum of correct votes > Q, False otherwise.
    """
    if weights is None:
        weights = [1] * len(expert_p)
    
    if Q is None:
        Q = np.sum(weights) / 2
    
    # Draw a binary outcome (0 or 1) for each expert based on their success probability
    outcomes = np.random.rand(len(expert_p)) <= expert_p
    total_correct = np.sum( np.dot(outcomes, weights))
    
    return total_correct > Q

