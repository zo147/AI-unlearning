from model_setup import setup_and_prepare
from ablation_clean import train_unlearning
from benchmarking import run_benchmark_df
from pandas import read_csv
import os
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device #CUDA
print(device)

model_name = "meta-llama/Meta-Llama-3.1-8B"  # Change model name accordingly
model, tokenizer, dataloader_t, dataloader_s = setup_and_prepare(model_name)
# run_benchmark_df(f'{model_name.split("/")[-1]}_BASELINE', read_csv(os.path.join('data', 'test_qa.csv')), model, tokenizer, device)

LEARNING_RATE = 2e-4  
EPOCHS = 22  # with batches of 2, for 1k batch passes we need 22 epochs
E1 = 0.1
E2 = None
E3 = 1
THRESHOLD = 5

model = train_unlearning(model, dataloader_t, dataloader_s, E1, E2, E3, LEARNING_RATE, EPOCHS, THRESHOLD)


run_benchmark_df(f'{model_name.split("/")[-1]}_{EPOCHS}epochs', read_csv(os.path.join('data', 'test_qa.csv')), model, tokenizer, device)


