import torch
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F

from model_setup import setup_and_prepare 

def unlearn(model, optimizer, accelerator, dataloader_t, dataloader_s, e1=1, e2=1, e3=4, epochs=1, threshold=5):
    # Step 1: Initialize any temporary variables
    baseline = []  # To store the pre-unlearning model's outputs for dataloader_s
    
    model.eval()
    print("Storing baseline predictions for safe data...")
    # Step 2: Get the model's predictions on dataloader_s (safe data) and store them as logits in baseline
    for batch in dataloader_s:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
        baseline.append(logits)
    
    print("Baseline predictions stored. Starting unlearning process...")
    
    # Step 3: Initialize unlearning loop
    
    # history =  [(0, 0)]
    for epoch in range(epochs):
        step_counter = 0
        # epoch_target_loss = 0
        # epoch_kl_loss = 0
        for target_batch, safe_batch in zip(dataloader_t, dataloader_s):
            # Move batches to device
            target_batch = {k: v.to(accelerator.device) for k, v in target_batch.items()}
            safe_batch = {k: v.to(accelerator.device) for k, v in safe_batch.items()}
            
            # Step 3.1: Perform first component (target_loss: gradient ascent on target data)
            model.train()
            target_outputs = model(input_ids=target_batch['input_ids'], attention_mask=target_batch['attention_mask'], labels=target_batch['input_ids'])
            target_loss = -1 * target_outputs.loss * e1 # Multiply by -1 to perform gradient ascent
            
            optimizer.zero_grad()
            if target_outputs.loss <= threshold:
                accelerator.backward(target_loss)  # Backpropagation for gradient ascent
                optimizer.step()
            
            # Step 3.2: Perform third component (KL loss: gradient descent on safe data)
            model.eval()
            current_safe_preds = model(safe_batch['input_ids'], attention_mask=safe_batch['attention_mask']).logits

            # Compare current model's safe data predictions to the baseline using KL divergence
            baseline_safe_preds = baseline[step_counter]  # Access corresponding baseline prediction
            kl_loss = F.kl_div(F.log_softmax(current_safe_preds, dim=-1), F.softmax(baseline_safe_preds, dim=-1), reduction='batchmean') * e3

            model.train()
            optimizer.zero_grad()  # Reset gradients before the next backprop
            accelerator.backward(kl_loss)  # Backpropagation for gradient descent
            optimizer.step()

            # Update step counter
            step_counter += 1
            # print(step_counter, end='\r')
            print(f'Step: {step_counter} done. Target Loss: {target_loss.cpu().detach().numpy()/e1} KL Loss: {kl_loss.cpu().detach().numpy()/e3}')
            # history.append((target_loss/e1, kl_loss/e3))
            # epoch_target_loss += target_loss.to('cpu') /e1
            # epoch_kl_loss += kl_loss.to('cpu') /e3

        # history.append((epoch_target_loss/step_counter, epoch_kl_loss/step_counter))
        print(f"Epoch {epoch+1}/{epochs} completed")

    print("Unlearning process complete.")
    return model


# Function to initiate the optimizer and prepare for unlearning
def train_unlearning(model, dataloader_t, dataloader_s, e1=1, e2=0, e3=4, learning_rate=1e-3, epochs=1, threshold=5):
    # Prepare optimizer and schedule
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = accelerator.prepare(optimizer)

    # Run the unlearning process
    model = unlearn(model, optimizer, accelerator, dataloader_t, dataloader_s, e1=e1, e2=e2, e3=e3, epochs=epochs)
    return model

if __name__ == "__main__":
    # Example of running the unlearning process
    # Model and data path are defined here for flexibility
    model_name = "meta-llama/Meta-Llama-3.1-8B"  # Change model name accordingly

    # Prepare model and data
    model, tokenizer, dataloader_t, dataloader_s = setup_and_prepare(model_name)

    learning_rate = 1e-3  # You can modify this
    epochs = 2  # Set your desired number of epochs

    model, history = train_unlearning(model, dataloader_t, dataloader_s, learning_rate, epochs)
    print("Unlearning complete!")