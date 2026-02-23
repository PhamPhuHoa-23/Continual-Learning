import argparse
from config.variables import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="yaml/config_zoom_1.yml",
                    help='Path to the configuration YAML file')
args = parser.parse_args()
load_config(args.config_path)

from algorithms.moe_base import PseudoExpert
from llm.llm_moe_base import LLMExpert

from config.variables import *
from algorithms.maj_voting import *
import pickle
from algorithms.zooming_algo import zooming_bandit
from llm.etl_data import parse_data_from_file

# Initialize experts with varying capabilities
if LLM_DATA_FILEPATH is not None:
    hist_records, llm_experts_names = parse_data_from_file(LLM_DATA_FILEPATH)
    if len(EXPERT_LLM_LIST) > 0:
        llm_experts_names = EXPERT_LLM_LIST

    experts = [
        LLMExpert(expert_name, hist_records, id=idx, p=EXPERT_P[idx]) 
        for idx, (expert_name, other_value) in enumerate(zip(llm_experts_names, EXPERT_P))
    ]
else:
    experts = [PseudoExpert(perf, idx) for idx, perf in enumerate(EXPERT_P)]

print("Running Zooming Theta ...")
print(f"{'Round':<6} {'Experts':<20} {'Selected':<10} {'Outcome':<8} {'Estimates (with CIs)':<40}")
print("-" * 90)

llm_play_bool = LLM_DATA_FILEPATH is not None
side_ordinal_info = SIDE_ORDINAL_INFO
zooming_dic = zooming_bandit(experts, 
                             time_steps=T, 
                             t_burn_in=T_BURN_IN, 
                             llm_expert_set=experts,
                             side_ordinal_info=side_ordinal_info)
print("theta_weights_est:", zooming_dic['theta_weights_est'])

with open(f"saved_experiments/{CONFIG_NAME}.pkl", "wb") as f:
    pickle.dump(zooming_dic, f)
