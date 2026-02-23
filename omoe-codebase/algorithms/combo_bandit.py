import numpy as np
from algorithms.maj_voting import *
from utils.helper_functions import expert_printout
from itertools import combinations
import random

def powerset(lst):
    return [
        subset 
        for r in range(1, len(lst) + 1) 
        for subset in combinations(lst, r)
    ]


class UCB_Bandit:
    def __init__(self, expert_subsets):
        self.subsets = expert_subsets
        self.num_arms = len(expert_subsets)
        self.counts = np.zeros(self.num_arms)  # Number of pulls per arm
        self.values = np.zeros(self.num_arms)  # Empirical mean reward per arm

    def select_arm(self, t):
        # Initialize all arms if not yet pulled
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Compute UCB for all arms
        ucb_values = [
            self.values[arm] + np.sqrt(2 * np.log(t) / self.counts[arm])
            for arm in range(self.num_arms)
        ]
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        # Update counts and values
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        # Update empirical mean incrementally
        self.values[chosen_arm] = ((n - 1) * self.values[chosen_arm] + reward) / n

def combo_bandit(experts, 
                 time_steps=T, 
                 t_burn_in=T_BURN_IN, 
                 opt_solution=None, 
                 llm_play_bool=False,
                 k_arm_size=-1):
    current_experts = experts.copy()

    if opt_solution is None:
        best_sublist, opt_solution = find_opt_egal_voting([expert.p for expert in experts])
    
    ucb_succes_count = 0
    lcb_success_count = 0
    success_count = 0
    empirical_reward_tracker = []
    regret_tracker = []
    ucb_tracker = []
    lcb_tracker = []
    
    if 0 < k_arm_size < len(current_experts):
        expert_subsets = list(combinations(current_experts, k_arm_size))
    else:
        expert_subsets = powerset(current_experts)

    bandit = UCB_Bandit(expert_subsets)
    
    for t in range(1, time_steps+1):
        if t <= t_burn_in:
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
        
        # Display current status
        estimates = []
        for ex in current_experts:
            est = ex.get_estimate()
            lcb = ex.get_lcb()
            ucb = ex.get_ucb()
            estimates.append(f"E{ex.id}: {est:.2f} ({lcb:.2f},{ucb:.2f})")
        
        # Continuously check for elimination opportunities
        if t > t_burn_in:
            
            chosen_arm = bandit.select_arm(t)
            chosen_expert_subset = expert_subsets[chosen_arm]
            arm_expert_p = [e.p for e in chosen_expert_subset]
            reward = empirical_is_majority_win(arm_expert_p)
            bandit.update(chosen_arm, reward)
            
            expert_printout(chosen_expert_subset)
            print(f"{t:<6} {str([ex.id for ex in chosen_expert_subset]):<20} "
            f"E{ex.id:<9} {str(outcome):<8} {' | '.join(estimates)}")
            
            est_ucbs = [e.get_ucb() for e in chosen_expert_subset]
            ucb_is_success = empirical_is_majority_win(est_ucbs)
            if ucb_is_success:
                ucb_succes_count += 1
            p_maj_ucb = ucb_succes_count/t
            ucb_tracker.append(p_maj_ucb)

            
            est_lcbs = [e.get_lcb() for e in chosen_expert_subset]
            lcb_is_success = empirical_is_majority_win(est_lcbs)
            if lcb_is_success:
                lcb_success_count += 1
            p_maj_lcb = lcb_success_count/t
            lcb_tracker.append(p_maj_lcb)

            if llm_play_bool:
                arm_expert_p = [e.play() for e in chosen_expert_subset] # use empirical estimates
            
            is_success = empirical_is_majority_win(arm_expert_p)
            if is_success:
                success_count += 1

            empirical_reward = success_count/t
            empirical_regret = opt_solution - empirical_reward
            
            empirical_reward_tracker.append(empirical_reward)
            regret_tracker.append(empirical_regret)
            
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
    
    return_dic = {
        "bandit": bandit,
        "expert_subsets": expert_subsets,
        "current_experts": current_experts,
        "empirical_reward_tracker": empirical_reward_tracker,
        "regret_tracker": regret_tracker,
        "ucb_tracker": ucb_tracker,
        "lcb_tracker": lcb_tracker,
        "opt_solution": opt_solution,
        "success_count": success_count,
        "original_experts": experts,
    }

    return return_dic
