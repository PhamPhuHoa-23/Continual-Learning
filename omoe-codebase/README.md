# LLM-EXP: Large Language Model Experimentation Platform

![Research](https://img.shields.io/badge/Level-Research-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

LLM-EXP is a sophisticated research platform for experimenting with different algorithms in the context of Large Language Model (LLM) mixture of experts modelling. The package implements several approaches for combining expert LLM outputs, including Successive Expert Elimination (SEE), Weighted Majority Voting (WMV), and traditional bandit algorithms.

## Features

- **Multiple Algorithm Implementations**:
  - Successive Expert Elimination (SEE)
  - Weighted Majority Voting (WMV)
  - Combinatorial Bandit (CB)
  - Zooming Algorithm
- **Live LLM Integration**: Connect with local LLMs via Ollama
- **Benchmarking**: Compare different approaches to expert combination
- **Flexible Configuration**: YAML-based configuration for easy experimentation

Install the required dependencies:
```
pip install -r requirements.txt
```

(Optional) For live LLM experiments, install Ollama and download the required models as needed. For example:

```
ollama pull samantha-mistral
ollama pull notus
ollama pull mistral
ollama pull gemma-7b
ollama pull deepseek-r1-14b
```

### Gurobipy

Please ensure `Gurobi 9.5.2` is installed on your machine, we use `gurobipy 11.0.0` to solve our mixed integer programs for weighted majority voting.

## Usage 

### Configuration

Modify `demo_config.yml` to configure your experiments. 


To run the Successive Expert Elimination (SEE) on Bernoulli bandits

```
python run_see.py --config_path yaml/config_see_1.yml
```

To run Weighted Majority Voting (WMV) 

```
run_wmv.py --config_path yaml/config_wmv_1.yml
```

For other baseline algorithms (combinatorial UCB or Zooming algorithm):

```
run_cb.py --config_path yaml/config_cb_1.yml
run_zoom.py --config_path yaml/comfig_zoom_1.yml
```

### LLM Expert Usage

Set up ollama by downloading all of the required models i.e.
'samantha-mistral', 'deepseek-r1-14b' etc. To run experiments with live LLM queries, uncomment the relevant fields in `demo_config.yml`, i.e. `hugging_face_data_name: 'tau/commonsense_qa'`, and run any of the algorithm scripts as shown above.

```
run_see.py --config_path yaml/demo_config.yml
run_wmv.py --config_path yaml/demo_config.yml
run_cb.py --config_path yaml/demo_config.yml
run_zoom.py --config_path yaml/demo_config.yml
```

Each algorithm will intereact with the LLM to sample questions from the Huggingface dataset, and score whether the answer is correct or not. Each algorithm will combine expert LLM outputs using the specified algorithm.


The system will:

- Sample questions from the specified Hugging Face dataset.
- Query each LLM for responses.
- Score the correctness of answers.
- Combine expert outputs using the selected mixture-of-experts algorithm.








