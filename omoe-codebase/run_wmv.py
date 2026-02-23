import argparse
from config.variables import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="yaml/config_wmv_1.yml",
                    help='Path to the configuration YAML file')
args = parser.parse_args()
load_config(args.config_path)

from algorithms.moe_base import PseudoExpert
from llm.llm_moe_base import LLMExpert

from config.variables import *
from algorithms.maj_voting import *
import pickle
from algorithms.theta_wmv import theta_mwv
from utils.helper_functions import expert_printout
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

print("Running theta WMV based on UCB bounds...")
print(f"{'Round':<6} {'Experts':<20} {'Selected':<10} {'Outcome':<8} {'Estimates (with CIs)':<40}")
print("-" * 90)

llm_play_bool = LLM_DATA_FILEPATH is not None
wmv_dic = theta_mwv(experts, time_steps=T, t_burn_in=T_BURN_IN, llm_play_bool=llm_play_bool)

current_experts = wmv_dic['current_experts']
expert_printout(current_experts)

with open(f"saved_experiments/{CONFIG_NAME}.pkl", "wb") as f:
    pickle.dump(wmv_dic, f)

