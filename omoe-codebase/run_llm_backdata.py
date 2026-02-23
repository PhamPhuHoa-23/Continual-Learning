from datasets import load_dataset
from llm.llm_query import *
from tqdm import tqdm
from config.variables import HUGGING_FACE_DATA_NAME
import random

# ds_read = load_dataset("openai/gsm8k", "socratic")
# llm_formatter_input = format_for_llm_gsm8k
# llm_query_func_input = get_short_answer_from_llm_gsm8k

ds_read = load_dataset("tau/commonsense_qa")
llm_formatter_input = format_for_llm_commonsense
llm_query_func_input = get_short_answer_from_llm_commonsense

# ds_read = load_dataset("google/boolq")
# llm_formatter_input = format_for_llm_boolq
# llm_query_func_input = get_short_answer_from_llm_boolq

if HUGGING_FACE_DATA_NAME is not None:
    huggingface_data_name = HUGGING_FACE_DATA_NAME
else:
    huggingface_data_name = "tau/commonsense_qa"

if huggingface_data_name == "openai/gsm8k":
    ds_read = load_dataset("openai/gsm8k", "socratic")
elif huggingface_data_name == "tau/commonsense_qa":
    ds_read = load_dataset("tau/commonsense_qa")
elif huggingface_data_name == "google/boolq":
    ds_read = load_dataset("google/boolq")
else:
    raise ValueError(f"Unsupported dataset: {huggingface_data_name}")

BACKDATA_PATH = 'saved_experiments/model_comparison_results_commonsenseqa_strong_chunk1.txt'

def run_weak_models(ds, llm_query_func=get_short_answer_from_llm_gsm8k, llm_formatter=format_for_llm_gsm8k):
    with open(BACKDATA_PATH, 'w') as f:
        for id in tqdm(range(1000, 2000), desc="Processing models", unit="item"):
            formatted_query_dic = llm_formatter(ds['train'][id])
            correct_answer = formatted_query_dic['correct_answer']

            # phi4_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "phi4"})
            # granite_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "granite-code"})
            # orca_mini_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "orca-mini"})
            # nous_hermes_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "nous-hermes"})
            
            mistral_openorca_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "mistral-openorca"})
            notus_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "notus"})
            samantha_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "samantha-mistral"})
            aya_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "aya"})
            
            
            # output_text = f"""ID:{id} | Correct:{correct_answer} | granite-code:{granite_answer} | phi4:{phi4_answer} | mistral-openorca:{mistral_openorca_answer} | notus:{notus_answer} | nous-hermes:{nous_hermes_answer} | samantha-mistral:{samantha_answer} | aya:{aya_answer} | orca-mini:{orca_mini_answer}"""
            output_text = f"""ID:{id} | Correct:{correct_answer} | mistral-openorca:{mistral_openorca_answer} | notus:{notus_answer} | samantha-mistral:{samantha_answer} | aya:{aya_answer}"""
            
            
            print(output_text)
            f.write(output_text + "\n")

def run_strong_models(ds, llm_query_func=get_short_answer_from_llm_gsm8k, llm_formatter=format_for_llm_gsm8k):
    with open(BACKDATA_PATH, 'w') as f:
        for id in tqdm(range(1000, 2000), desc="Processing models", unit="item"):
            formatted_query_dic = llm_formatter(ds['train'][id])
            correct_answer = formatted_query_dic['correct_answer']

            # Strong LLMs (replacing the weaker ones)
            gemma_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "gemma:7b"})
            qwen_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "qwen:14b"})
            deepseek_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "deepseek-r1:14b"})
            mistral_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "mistral:7b"})
            phi4_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "phi4"})
            # olmo_answer ='} llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "olmo2:13b"})
            # mixtral_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "mixtral:8x7b"})
            # llava_answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": "llava:13b"})
            
            formatted_question = llm_formatter(ds['train'][id])
            
            print(f"Question: {formatted_question}")
            output_text = f"""ID:{id} | Correct:{correct_answer} | qwen-14b:{qwen_answer} | deepseek-r1-14b:{deepseek_answer} | gemma-7b:{gemma_answer} | mistral:{mistral_answer} | phi4:{phi4_answer}"""
            
            # print(f"Question: {ds_commonformatted_question}")
            # output_text = f"""ID:{id} | Correct:{correct_answer} | 
            # gemma-7b:{gemma_answer} | 
            # qwen-14b:{qwen_answer} | 
            # deepseek-r1-14b:{deepseek_answer} | 
            # phi4:{phi4_answer} | 
            # mistral:{mistral_answer} | 
            # olmo2-13b:{olmo_answer} | 
            # mixtral-8x7b:{mixtral_answer}"""
            
            print(output_text)
            f.write(output_text + "\n")

def run_live_query(llm_query_func=get_short_answer_from_llm_gsm8k, 
                   llm_formatter=format_for_llm_gsm8k,
                   llm_model_name="phi4"):
    
    id = random.sample(range(0, len(ds_read)), 1)[0]

    formatted_query_dic = llm_formatter(ds_read['train'][id])
    correct_answer = formatted_query_dic['correct_answer']
    
    answer = llm_query_func(**{"query_dic": formatted_query_dic, "llm_model": llm_model_name})
    
    if str(answer).lower() == str(correct_answer).lower():
        print(f"Correct! {answer} == {correct_answer}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    run_weak_models(ds=ds_read, 
                      llm_query_func=llm_query_func_input, 
                      llm_formatter=llm_formatter_input)

