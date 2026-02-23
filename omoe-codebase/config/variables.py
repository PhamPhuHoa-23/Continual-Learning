# config/variables.py
import yaml

# Default values (can be overridden by load_config)
CONFIG_PATH = 'yaml/config_zoom_3.yml'
CONFIG_NAME = CONFIG_PATH.split('/')[-1].split('.')[0]
config = {}
EXPERT_P = None
T_BURN_IN = 100
T = 1000
DELTA_CONFIDENCE = 0.1
ALPHA_LEARN = 0.5
ALPHA_DISPLAY = 0.05
LP_EPSILON = 1e-2
THETA_LB = 0.001
VOTING_EPSILON = 0.0
DELAY_LP_INTERVAL = 1
LLM_DATA_FILEPATH = None
EXPERT_LLM_LIST = []
K_ARM_SIZE = -1
SIDE_ORDINAL_INFO = True
HUGGING_FACE_DATA_NAME = None  # Default dataset name

def load_config(config_path=None):
    global CONFIG_PATH, CONFIG_NAME, config
    global EXPERT_P, T_BURN_IN, T, DELTA_CONFIDENCE, ALPHA_LEARN
    global ALPHA_DISPLAY, LP_EPSILON, THETA_LB, VOTING_EPSILON, DELAY_LP_INTERVAL
    global LLM_DATA_FILEPATH, EXPERT_LLM_LIST, K_ARM_SIZE, SIDE_ORDINAL_INFO, HUGGING_FACE_DATA_NAME
    
    if config_path:
        CONFIG_PATH = config_path
    CONFIG_NAME = CONFIG_PATH.split('/')[-1].split('.')[0]

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    EXPERT_P = config['expert_p']
    T_BURN_IN = config['t_burn_in']
    T = config['t']
    DELTA_CONFIDENCE = config['delta_confidence']
    ALPHA_LEARN = config['alpha_learn']
    ALPHA_DISPLAY = config.get('alpha_display', 0.05)
    LP_EPSILON = float(config.get('lp_epsilon', 1e-2))
    THETA_LB = float(config.get('theta_lb', 0.001))
    VOTING_EPSILON = float(config.get('voting_epsilon', 0.0))
    DELAY_LP_INTERVAL = config.get('delay_lp_interval', 1)
    LLM_DATA_FILEPATH = config.get('llm_data_filepath', None)
    EXPERT_LLM_LIST = config.get('expert_llm_list', [])
    K_ARM_SIZE = config.get('k_arm_size', -1)
    SIDE_ORDINAL_INFO = config.get('side_ordinal_info', True)
    HUGGING_FACE_DATA_NAME = config.get('hugging_face_data_name', None)
    pass
