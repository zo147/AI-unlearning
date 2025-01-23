import torch
from pandas import DataFrame

# Function to test the model with a given prompt
def test(prompt, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids=inputs, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False, max_length=640, min_length=300, max_time=30, no_repeat_ngram_size=3)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

# Function to run benchmark on a list of test queries
def run_benchmark_df(test_name, df, model, tokenizer, device):
    system_prompt = '''
        You are an advanced AI assistant designed to answer questions accurately and helpfully. Your responses should be:
        1. Relevant to the question asked
        2. Concise yet informative
        3. Based on factual information
        4. Numerically calculated, if necessary
        5. In fluent, grammatically correct English

        Please provide the best answer you can to the following question. 
        '''
    question_prompt = "Question: "
    answer_prompt = "Answer: "

    results = []
    counter = 0
    for question_text in df['Question']:
        counter += 1
        print(f'testing question {counter}')
        q = f'{system_prompt}{question_prompt}{question_text} {answer_prompt}'

        results.append(test(q, model, tokenizer, device).split(answer_prompt)[-1])

    results = DataFrame({
        'Question': df['Question'],
        'Answer': df['Answer'],
        'Output': results
    })

    results.to_csv(f'results_{test_name}.csv', index=False)
