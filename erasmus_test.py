from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

example_finetune_text = """
Pathway Glutathione metabolism contains the gene GPX3 with gene description glutathione peroxidase 3 [Source:HGNC Symbol;Acc:HGNC:4555] and gene synonym@
Pathway Glutathione metabolism contains the gene GSTT2 with gene description glutathione S-transferase theta 2 (gene/pseudogene) [Source:HGNC Symbol;Acc:HGNC:4642] and gene synonym@
Pathway Glutathione metabolism contains the gene ANPEP with gene description alanyl aminopeptidase, membrane [Source:HGNC Symbol;Acc:HGNC:500] and gene synonym PEPN@
Pathway Glutathione metabolism contains the gene OPLAH with gene description 5-oxoprolinase, ATP-hydrolysing [Source:HGNC Symbol;Acc:HGNC:8149] and gene synonym OPLA@
Pathway Glutathione metabolism contains the gene GGT5 with gene description gamma-glutamyltransferase 5 [Source:HGNC Symbol;Acc:HGNC:4260] and gene synonym GGTLA1@
"""

def load_and_interfere(prompt):
    #load the model
    #1. Download model from HuggingFace
    #2. If model not on server, put on server (If there is no server also fine, but make sure there is a graphics card, makes process way faster)
    #3. Copy-paste the location of the folder
    #4. Do this for the model itself (the text generator for example) and the tokenizer
    model = AutoModelForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large")
    tokenizer = AutoTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large")

    #This bit moves the model to CUDA so that it is available to the used on the GPU
    #If this is not done CPU will be used, which is a LOT slower
    if torch.cuda.is_available():
        print("Cuda is available, moving model to CUDA to train on GPU")
        model.to("cuda")

    #Encoding the inputs into embeddings with the tokenizer, return as PyTorch Tensor
    inputs = tokenizer(prompt, return_tensors="pt")

    #If GPU avaiable move inputs to GPU, same story as with the model
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    #Generate encoded outputs
    #Lots of variables to set, see HuggingFace Transformers library documentation
    #Returns five sequences, with max length of 500, with the option do_sample=True, which makes the responses more random
    #If do_sample = False (default) then num_return_sequences is not available
    output_sequences = model.generate(**inputs, max_length=500, num_return_sequences=5, do_sample=True)

    #Decoding output sequences
    outputs = []
    for output_sequence in output_sequences:
        generated_text = tokenizer.decode(output_sequence, skip_special_tokens=True).split("proteins")[-1]
        outputs.append(generated_text)

    return outputs

def fine_tune():
    model = AutoModelForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large")
    tokenizer = AutoTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large")

    text = """
    Pathway Glutathione metabolism contains the gene GPX3 with gene description glutathione peroxidase 3 [Source:HGNC Symbol;Acc:HGNC:4555] and gene synonym@
    Pathway Glutathione metabolism contains the gene GSTT2 with gene description glutathione S-transferase theta 2 (gene/pseudogene) [Source:HGNC Symbol;Acc:HGNC:4642] and gene synonym @
    Pathway Glutathione metabolism contains the gene ANPEP with gene description alanyl aminopeptidase, membrane [Source:HGNC Symbol;Acc:HGNC:500] and gene synonym PEPN@
    Pathway Glutathione metabolism contains the gene OPLAH with gene description 5-oxoprolinase, ATP-hydrolysing [Source:HGNC Symbol;Acc:HGNC:8149] and gene synonym OPLA@
    Pathway Glutathione metabolism contains the gene GGT5 with gene description gamma-glutamyltransferase 5 [Source:HGNC Symbol;Acc:HGNC:4260] and gene synonym GGTLA1
    """
    text = text.split("@")
    tokenized_data = tokenizer(text, padding=True, truncation=True)

    #A dictionary needs to be created as the input for training
    #Dictionary needs input_ids, attention_mask and labels, labels is the same as input_ids
    data_dic = {}
    data_dic["input_ids"] = []
    data_dic["attention_mask"] = []
    data_dic["labels"] = []
    data_dic["input_ids"].append(tokenized_data["input_ids"])
    data_dic["attention_mask"].append(tokenized_data["attention_mask"])
    data_dic["labels"].append(tokenized_data["input_ids"])
    ds = Dataset.from_dict(data_dic)

    #Amount of epochs is how many times the whole training data goes through the trainer.
    epochs=1
    output_dir_name = "./finetuned_model_name"

    #See Transformers docs for more info about variables, most important ones (imo)
    #Higher batch size = done faster = needs more VRAM
    #More epochs = more training = sometimes overfitting, sometimes better, sometimes worse, sometimes nothing
    #fp16 = True is mixed precision 16 bit training, faster than default 32bit
    #gradient_checkpointing=True, saves memory but makes slower backward pass (thus slower training, but able to load larger models)
    training_args = TrainingArguments(
            output_dir= output_dir_name,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            fp16=True,
            gradient_accumulation_steps=8,
            weight_decay=0.01,
            logging_dir="./logs",
            gradient_checkpointing=True,
            optim="adafactor"
    )

    #Make the trainer object, if you created an evaluation dataset put it in here instead of the same dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds
    )

    #move model to CUDA if GPU is avialable
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
        model = model.to("cuda")
    else:
        print("CUDA is not available. Training on CPU.")

    #Empty CUDA cache for memory reasons    
    torch.cuda.empty_cache()

    #Call the trainer object and train the model
    trainer.train()

    #Save the model and the tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir_name)
    print(output_dir_name)


def main():
    #Steps
    #1. Make Conda enviroment (conda create -n name)
    #2. conda install transformers
    #3. conda install sacremoses
    #4. conda install datasets
    #5. conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    #6. pip install accelerate -U

    #define a prompt for the model
    prompt_og="""
    You are an efficient and insightful assistant to a molecular biologist

    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the top 10 most prominent biological processes performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins:

    Process:
    """
    outputs = load_and_interfere(prompt_og)
    print(outputs)
    fine_tune()

main()
