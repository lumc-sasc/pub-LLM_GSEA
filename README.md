# Pub-LLM_GSEA
Repository for testing the performance of GPT4 and open source language models on a GSEA task

### Dependencies
Install Conda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
Create Conda environment
```
conda create -n run_models_sasc python=3.11.6
```
Install necessary packages
```
conda install -c conda-forge transformers=4.38.1
conda install -c conda-forge cuda=11.8.0
conda install -c conda-forge matplotlib=3.8.2
conda install -c numpy=1.24.4
conda install -c conda-forge openai=1.14.1
conda install -c conda-forge pytorch=2.1.2
conda install -c conda-forge sentence-transformers=2.5.1
conda install -c conda-forge datasets=2.14.7
csv
json
pandas
scipy
```

Download repository
```
git clone https://github.com/lumc-sasc/pub-LLM_GSEA.git
```
## Usage
Set OPENAI API Key in BASH:
```
echo "export OPENAI_API_KEY=<PLACEHOLDER>" >> ~/.zshrc

source ~/.zshrc

echo $OPENAI_API_KEY
```
Activate Conda environment:
```
conda activate run_models_sasc
```

### Generating GPT4 ground truth and testing consistency of prompts using API

`generate_gpt4.py`

In function main

Define which path the script takes by setting the variable compare
```
compare = 1
- tests consistency of prompts, does all calculations
compare = 2
- makes ground truth set
compare = 3
- makes a plot for the scores calculated when compare = 1
```
Prompt to be used for making the ground truth data set can be set in function `make_truthset`
```
prompt = """
    You are an efficient and insightful assistant to a molecular biologist

    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the top 10 most prominent biological processes performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
  """
```

Prompts and genes to be used for testing consistency of prompts can be set in function `vars`
```
nested_list: nested list of gene sets.
content_list_removed: a dictionary where the key is the indicator of the prompt and the value is the prompt.
```


### Testing performance of local models

`testing_multi.py`

In function variables(), a dictionary gets returned with all the models that get tested
One such dictionary looks like this:
```
    models1={
        "BioGPT-large": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large",
        "PubMedQA": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large-PubMedQA",
        "biogpt": "/exports/sascstudent/svanderwal2/programs/test_new_models/biogpt",
        "BioMedLM": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioMedLM",
        "BioMedGPT": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioMedGPT-LM-7B"
    }
```
Every key is the name of the model tested, and the value the path of the model.
So if a new model is fine-tuned, add this model with path to an existing dictionary, or create a new one.
In this function, the prompt and gene sets used for testing are also set:
```
prompt_og: string variable which contains the prompt
nested_list: nested list of genes which contains the genes tested
```
This function takes the variable `test` which is set in `main`, this dictates which dictionary gets returned.

In function main, the variable `test` can be set, which dictates which path the script takes.
This variable is a list into which multiple or one string(s) can be put: `5base, 8structures, custom_onegene_morepaths or custom_pathgenedesc`

This script also generates two plots for visually inspecting performance of local models
```
file_line = "line_all.png"
title_line = "Performance of 5 local models on a GSEA task, 1 iterations"
file_bar = "bar_all.png"
title_bar = "Performance of 5 local models on a GSEA task, 1 iterations"
```

### Generate training data

### Fine-tuning a model
