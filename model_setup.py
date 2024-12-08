import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, set_seed, BitsAndBytesConfig
while True:
  try:
    from transformers import DataCollatorWithPadding
    break
  except:
    continue
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
import numpy as np
import glob

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device
print(f"Device: {device}")

# Global seed setting for reproducibility
torch.manual_seed(11)
np.random.seed(11)
set_seed(11)

# Function to load model and tokenizer
def load_model(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token #FOR LLAMA ONLY
    
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05
    )

    # while True:
    #     try:
    #         raw_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True
    #                                                         , attn_implementation="eager", quantization_config=config) #comment out for non-Gemma ?
    #         break
    #     except:
    #         continue
    raw_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True
                                                            , attn_implementation="eager", quantization_config=config)
    
    raw_model = prepare_model_for_kbit_training(raw_model)
    model = get_peft_model(raw_model, lora_config)
    model = model.to(device)
    print(model.print_trainable_parameters())        
    return model, tokenizer

# Function to load text data for ablation (text files)
def load_ablation_data():
    targets = []
    for i in glob.glob("data/target*.txt"):
        with open(i) as t:
            plaintext_t = t.read()
            targets.append(plaintext_t)

    safe = []
    for i in glob.glob("data/safe.txt"):
        with open(i) as s:
            plaintext_s = s.read()
            safe.append(plaintext_s)

    raw_ds_t = Dataset.from_dict({'text': [targets]})
    raw_ds_s = Dataset.from_dict({'text': [safe]})
    
    return raw_ds_t, raw_ds_s

def preprocess(examples, tokenizer):
  return tokenizer(
      examples["text"],
      max_length=128,
      padding="max_length",
      truncation=True,
      stride=10,
      # return_tensors="pt",
      return_overflowing_tokens=True
  )

# Function to prepare data for unlearning
def prepare_data(raw_ds_t, raw_ds_s, tokenizer, device):
    ds_t_nested = raw_ds_t.map(preprocess, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer}).remove_columns('overflow_to_sample_mapping')
    ds_s_nested = raw_ds_s.map(preprocess, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer}).remove_columns('overflow_to_sample_mapping')

    ds_t = Dataset.from_dict({'input_ids': ds_t_nested['input_ids'][0], 'attention_mask': ds_t_nested['attention_mask'][0]})
    ds_s = Dataset.from_dict({'input_ids': ds_s_nested['input_ids'][0][:len(ds_t_nested['input_ids'][0])], 'attention_mask': ds_s_nested['attention_mask'][0][:len(ds_t_nested['input_ids'][0])]})
    
    dc = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    dataloader_t = torch.utils.data.DataLoader(ds_t, batch_size=2, collate_fn=dc)
    dataloader_s = torch.utils.data.DataLoader(ds_s, batch_size=2, collate_fn=dc)
    return dataloader_t, dataloader_s


# Main function to prepare model and data
def setup_and_prepare(model_name):
    model, tokenizer = load_model(model_name)
    raw_ds_t, raw_ds_s = load_ablation_data()
    dataloader_t, dataloader_s = prepare_data(raw_ds_t, raw_ds_s, tokenizer, device)
    model, tokenizer, dataloader_t, dataloader_s = accelerator.prepare(model, tokenizer, dataloader_t, dataloader_s)
    return model, tokenizer, dataloader_t, dataloader_s

if __name__ == "__main__":
    # Model and data path are defined here for flexibility
    model_name = "meta-llama/Meta-Llama-3.1-8B"  # Change model name accordingly

    # Prepare model and data
    model, tokenizer, dataloader_t, dataloader_s = setup_and_prepare(model_name)
    print("Setup complete!")