import numpy as np
from algorithms.maj_voting import *
from utils.helper_functions import expert_printout
from algorithms.lp_solver import solve_lp_weights


def theta_mwv(experts, time_steps=T, t_burn_in=T_BURN_IN, opt_solution=None, delay_lp_interval=DELAY_LP_INTERVAL, llm_play_bool=False):
    current_experts = experts.copy()

    if opt_solution is None:
        # solve LP program
        expert_p = [expert.p for expert in experts]
        theta_weights = solve_lp_weights(expert_p)
        opt_solution = p_maj_vote(expert_p, weights=theta_weights)
        _, opt_solution_see = find_opt_egal_voting([expert.p for expert in experts])
    
    ucb_success_count = 0
    lcb_success_count = 0
    success_count = 0
    empirical_reward_tracker = []
    regret_tracker = []

    ucb_tracker = []
    lcb_tracker = []

    theta_weights_est = list(np.ones(len(expert_p)))
    theta_weights_ucb = list(np.ones(len(expert_p)))
    theta_weights_lcb = list(np.ones(len(expert_p)))

    for t in range(1, time_steps+1):
        
        if t <= t_burn_in:
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
        
        if t > t_burn_in:

            # Display current status
            estimates = []
            for ex in current_experts:
                est = ex.get_estimate()
                lcb = ex.get_lcb()
                ucb = ex.get_ucb()
                estimates.append(f"E{ex.id}: {est:.2f} ({lcb:.2f},{ucb:.2f})")
            
            print(f"{t:<6} {str([ex.id for ex in current_experts]):<20} "
                f"E{ex.id:<9} {str(outcome):<8} {' | '.join(estimates)}")

            expert_ucb_p = [e.get_ucb() for e in current_experts]
            
            if t%delay_lp_interval == 0:
                theta_weights_ucb = solve_lp_weights(expert_ucb_p)
            
            ucb_is_success = empirical_is_majority_win(expert_ucb_p, weights=theta_weights_ucb)
            if ucb_is_success:
                ucb_success_count += 1
            p_maj_ucb = ucb_success_count/t
            ucb_tracker.append(p_maj_ucb)

            expert_lcb_p = [e.get_lcb() for e in current_experts]
            
            if t%delay_lp_interval == 0:
                theta_weights_lcb = solve_lp_weights(expert_lcb_p)
            
            lcb_is_success = empirical_is_majority_win(expert_lcb_p, weights=theta_weights_lcb)
            if lcb_is_success:
                lcb_success_count += 1
            p_maj_lcb = lcb_success_count/t
            lcb_tracker.append(p_maj_lcb)

            expert_p = [e.get_estimate() for e in current_experts]
            if t%delay_lp_interval == 0:
                theta_weights_est = solve_lp_weights(expert_p)

            if llm_play_bool:
                emp_experts_p = [e.play() for e in current_experts] # use empirical estimates
            else:
                emp_experts_p = [e.p for e in current_experts]

            is_success = empirical_is_majority_win(emp_experts_p, weights=theta_weights_est)
            
            if t > 50:
                pass
            
            if is_success:
                success_count += 1

            empirical_reward = success_count/t
            empirical_regret = opt_solution - empirical_reward
            
            empirical_reward_tracker.append(empirical_reward)
            regret_tracker.append(empirical_regret)
            
            for expert in current_experts:
                outcome = expert.play()
                expert.update(outcome)
            
            expert_printout(current_experts)
    
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
