import numpy as np
from algorithms.maj_voting import *
from utils.helper_functions import expert_printout
from random import random

def egal_adv_function(subset1, subset2, reward_func=p_maj_vote):
    
    if not subset1 or not subset2:
        return 0  # No advantage if one subset is empty
    
    print("\nSubset 1:" + str([(expert.id, expert.get_estimate()) for expert in list(subset1)]))
    print("\nSubset 2:" + str([(expert.id, expert.get_estimate()) for expert in list(subset2)]))

    expert_competence_1 = [expert.get_lcb() for expert in subset1]
    p_cand = p_maj_vote(expert_competence_1)

    expert_competence_2 = [expert.get_ucb() for expert in subset2]
    p_retain = p_maj_vote(expert_competence_1 + expert_competence_2)

    print(f"p_cand: {p_cand}, p_retain: {p_retain}")

    p_adv = p_retain - p_cand

    return p_adv

def expert_subset_advantage_function(expert_list):
    if len(expert_list) <= 1:
        return expert_list.copy()
    
    # Sort experts by their empirical success probability
    sorted_experts = sorted(expert_list, key=lambda x: x.get_estimate())
    
    # Find the first significant gap in UCB bounds
    split_point = None
    for i in range(len(sorted_experts)-1):
        lower_ucb = sorted_experts[i].get_estimate() - sorted_experts[i].get_ucb()
        upper_ucb = sorted_experts[i+1].get_estimate() + sorted_experts[i+1].get_ucb()
        
        if lower_ucb > upper_ucb:  # Non-overlapping confidence intervals
            split_point = i
            break
    
    if split_point is None:
        return expert_list.copy()  # No clear split point found
    
    # Split into two groups
    group1 = sorted_experts[:split_point+1]
    group2 = sorted_experts[split_point+1:]
    
    # Compare groups using external advantage function
    advantage = egal_adv_function(group1, group2)
    
    if advantage >= 0:
        return expert_list.copy()  # Keep all experts
    else:
        return group1  # Eliminate the worse-performing group

def check_for_elimination(current_experts):
    """Checks for complete non-overlapping confidence bounds and performs elimination"""
    if len(current_experts) <= 1:
        return current_experts.copy()
    
    # Sort experts by their empirical success probability
    sorted_experts = sorted(current_experts, key=lambda x: x.get_estimate(), reverse=True)
    
    # Find all potential split points where there is complete disjointness
    split_points = []
    for i in range(1, len(sorted_experts)):
        # Group 1: experts [0..i-1]
        # Group 2: experts [i..n-1]
        
        # Find the minimum UCB in group 1
        cand_group = sorted_experts[:i]
        min_lcb_group1 = min(expert.get_lcb() for expert in cand_group)

        # Find the maximum LCB in group 2
        elim_group = sorted_experts[i:]
        max_ucb_group2 = max(expert.get_ucb() for expert in elim_group)
    
        if min_lcb_group1 > max_ucb_group2:  # Complete non-overlap
            split_points.append(i)
    
    if not split_points:
        return current_experts.copy()  # No elimination possible
    
    # Choose the largest split point (most experts in group 1)
    best_split = min(split_points)
    
    # Split into two groups
    group1 = sorted_experts[:best_split]
    group2 = sorted_experts[best_split:]
    
    # Compare groups using external advantage function
    advantage = egal_adv_function(group1, group2)
    
    if advantage < 0:  # group1 is better
        print(f"Eliminating experts {[e.id for e in group2]} based on complete UCB separation")
        return group1
    else:
        print(f"Retaining experts due to advantage: {advantage} for group {[e.id for e in current_experts]}")
        return current_experts.copy()  # No elimination

def successive_expert_elimination(experts, time_steps=T, t_burn_in=T_BURN_IN, opt_solution=None, llm_play_bool=False):
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
        
        print(f"{t:<6} {str([ex.id for ex in current_experts]):<20} "
            f"E{ex.id:<9} {str(outcome):<8} {' | '.join(estimates)}")
        
        # Continuously check for elimination opportunities
        if t > t_burn_in:
            expert_printout(current_experts)
            new_experts = check_for_elimination(current_experts)
            new_experts_p = [e.p for e in new_experts]
            
            est_ucbs = [e.get_ucb() for e in new_experts]
            if llm_play_bool:
                est_ucbs = [random() < p for p in est_ucbs] # use empirical estimates

            ucb_is_success = empirical_is_majority_win(est_ucbs)
            if ucb_is_success:
                ucb_succes_count += 1
            p_maj_ucb = ucb_succes_count/t
            ucb_tracker.append(p_maj_ucb)
            
            est_lcbs = [e.get_lcb() for e in new_experts]

            if llm_play_bool:
                est_lcbs = [random() < p for p in est_lcbs] # use empirical estimates

            lcb_is_success = empirical_is_majority_win(est_lcbs)

            if lcb_is_success:
                lcb_success_count += 1
            p_maj_lcb = lcb_success_count/t
            lcb_tracker.append(p_maj_lcb)

            if llm_play_bool:
                new_experts_p = [e.play() for e in new_experts] # use empirical estimates

            is_success = empirical_is_majority_win(new_experts_p)
            
            if is_success:
                success_count += 1

            empirical_reward = success_count/t
            empirical_regret = opt_solution - empirical_reward
            
            empirical_reward_tracker.append(empirical_reward)
            regret_tracker.append(empirical_regret)

            if len(new_experts) < len(current_experts):
                current_experts = new_experts
                if len(current_experts) == 1 and not llm_play_bool:
                    print(f"Early stopping at round {t} - only one expert remains")
                    break
            
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
    
    return_dic = {
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
