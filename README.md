# Pub-LLM_GSEA
Repository for testing the performance of GPT4 and open source language models on a GSEA task

## Workflow
Set OPENAI API Key in BASH:
```
echo "export OPENAI_API_KEY=<PLACEHOLDER>" >> ~/.zshrc

source ~/.zshrc

echo $OPENAI_API_KEY
```
Install Conda and create conda environment:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
conda --version
```
`conda 23.10.0`

Install all necassary packages



Generate GPT4 ground truth

Test performance of local models

Inspect plots

Make training data

Train models

Test performance again

Inspect plots

### Dependencies
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

```

## Usage
### Testing consistency of prompts with GPT API

`testing_multi.py`
In function variables()
```
return models1
```
Set where to save the plots created and which title they should have
```
file_line = "line_all.png"
title_line = "Performance of 5 local models on a GSEA task, 1 iterations"
file_bar = "bar_all.png"
title_bar = "Performance of 5 local models on a GSEA task, 1 iterations"
```

### Generating GPT4 ground truth using API

`generate_gpt.py`

The ground truth is saved in a dictionary and written away in a file set with:
```
file_save = "gpt_ground_truth.txt"
```

### Testing performance of local models
Set variable for the file which contains the ground truth
```
file_n = "/exports/sascstudent/svanderwal2/programs/test_new_models/gpt_ground_truth.txt"
```

This script generates a file to manually check result from local model vs ground truth
```
file_write = "testing_multi_result_gpt4_vs_local.csv"
```
This script also generates two plots for visually inspecting performance of local models
```
file_line = "line_all.png"
title_line = "Performance of 5 local models on a GSEA task, 1 iterations"
file_bar = "bar_all.png"
title_bar = "Performance of 5 local models on a GSEA task, 1 iterations"
```
Set how many iterations the programs will do on every gene set
```
iterations = 2
```
