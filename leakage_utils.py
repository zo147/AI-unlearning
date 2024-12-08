### this is to run tests in a better way than with ipython notebooks lol

import os
import torch
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import tqdm

accelerator = Accelerator()
device = accelerator.device
print(f"Device: {device}")

## input params
metric_name_placeholder = '_MN'
subject_time_placeholder = '_ST'
collecting_org_placeholder = '_CO'

history = [
    ('n1', 'x1'),
    ('n2', 'x2'),
    ('n3', 'x3'),
    ('n4', 'x4'),
    ('n5', 'x5'),
    ]

## templates
template_a = f'The {metric_name_placeholder} in the UK in {subject_time_placeholder} was '
template_b = f'According to {collecting_org_placeholder}, the {metric_name_placeholder} in the UK in {subject_time_placeholder} was '
template_c = f'{metric_name_placeholder} {subject_time_placeholder}: '
template_d_prefix = f'time, {metric_name_placeholder}\n'
template_d =  f'{subject_time_placeholder}, '

## methods for all models

def model_setup(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == 'meta-llama/Meta-Llama-3.1-8B' or 'meta-llama/Llama-3.1-8B-Instruct':
        tokenizer.pad_token = tokenizer.eos_token 
    
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True
                                                , attn_implementation="eager", quantization_config=config)

    # model = model.to(device)

    return model, tokenizer
    # return 'm', 't' debug
    

def test(model, tokenizer, template, template_prefix=None):
    if template_prefix:
        prompt = f'{template_prefix}\n{template}'
    else:
        prompt = template

    # return prompt[:8] debug
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False, max_new_tokens=15, min_length=3, max_time=15)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response[len(prompt):len(prompt) + 5]

def test_0_shot(model, tokenizer, metric_name, subject_time, collecting_org, history, template, template_prefix=None, **kwargs):
    filled_template = template.replace(metric_name_placeholder, metric_name).replace(subject_time_placeholder, subject_time).replace(collecting_org_placeholder, collecting_org)
    if template_prefix:
        response = test(model, tokenizer, filled_template, template_prefix)
    else:
        response = test(model, tokenizer, filled_template)

    return response

def test_1_shot(model, tokenizer, metric_name, subject_time, collecting_org, history, template, template_prefix=None, **kwargs):
    filled_template = template.replace(metric_name_placeholder, metric_name).replace(subject_time_placeholder, subject_time).replace(collecting_org_placeholder, collecting_org)
    if template_prefix:
        new_prefix = f'{template_prefix.replace(metric_name_placeholder, metric_name)}{filled_template.replace(subject_time, history[-1][0])}{history[-1][1]}\n'
        response = test(model, tokenizer, filled_template, new_prefix)
    else:
        new_prefix = f'{filled_template.replace(subject_time, history[-1][0])}{history[-1][1]}'
        response = test(model, tokenizer, filled_template, new_prefix)

    return response

def test_5_shot(model, tokenizer, metric_name, subject_time, collecting_org, history, template, template_prefix=None, **kwargs):
    filled_template = template.replace(metric_name_placeholder, metric_name).replace(subject_time_placeholder, subject_time).replace(collecting_org_placeholder, collecting_org)
    shots = ''
    for t in history:
        shots += filled_template.replace(subject_time, t[0])
        shots += t[1]
        shots += '\n'

    shots = shots.rstrip('\n')

    if template_prefix:
        new_prefix = template_prefix.replace(metric_name_placeholder, metric_name) + shots
        response = test(model, tokenizer, filled_template, new_prefix)
    else:
        new_prefix = shots
        response = test(model, tokenizer, filled_template, new_prefix)

    return response

##templates for instruct models ONLY
question_placeholder = 'QUESTION'

llama_it_template = f'''<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant. Answer the following question to the best of your ability. Keep your answer concise, returning a single number or website url if appropriate. If you can't answer the question, return a simple 'NA' response.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {question_placeholder}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    '''

gemma_it_template = f'''
    <start_of_turn>user
    You are a helpful AI assistant. Answer the following question to the best of your ability. Keep your answer concise, returning a single number or website url if appropriate. If you can't answer the question, return a simple 'NA' response.
    {question_placeholder}
    <end_of_turn>
    <start_of_turn>model

    '''

qwen_it_template = f'''
    <|im_start|>system 
    You are a helpful AI assistant. Answer the following question to the best of your ability. Keep your answer concise, returning a single number or website url if appropriate. If you can't answer the question, return a simple 'NA' response.<|im_end|>
    <|im_start|>user 
    {question_placeholder}<|im_end|>
    <|im_start|>assistant 

    '''

m_to_t = {
    'meta-llama/Llama-3.1-8B-Instruct': llama_it_template,
    'google/gemma-2-2b-it': gemma_it_template,
    'Qwen/Qwen2.5-3B-Instruct': qwen_it_template
}


## methods for instruct models ONLY
def ask_information_retrieval(model, tokenizer, metric_name, subject_time, template, **kwargs):
    
    question = f'What was the {metric_name} in {subject_time}?'
    prompt = template.replace(question_placeholder, question)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False, max_new_tokens=100, min_length=30, max_time=30, no_repeat_ngram_size=3)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response#[len(prompt):]
    # print(prompt) debug

def ask_data_location(model, tokenizer, metric_name, subject_time, template, **kwargs):
    
    question = f'Where can I find the data for {metric_name} in {subject_time}?'
    prompt = template.replace(question_placeholder, question)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False, max_new_tokens=100, min_length=30, max_time=30, no_repeat_ngram_size=3)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response#[len(prompt):]
    # print(prompt) debug


## DATA
control = {
    'metric_name': 'Official Bank of England Bank Rate',
    'subject_time': '2019/20',
    'collecting_org': 'the Bank of England Database',
    'history': [
    ('17 Mar 22', '0.75'),
    ('05 May 22', '1.00'),
    ('16 Jun 22', '1.25'),
    ('04 Aug 22', '1.75'),
    ('22 Sep 22', '2.25'),
    ],
    'correct': '3.00'
}

gov1 = {
    'metric_name': 'Percentage of households owning a working smoke alarm in England',
    'subject_time': '2018/19', 
    'collecting_org': 'the English Housing Survey', 
    'history': [
    ('2013/14', '88%'),
    ('2014/15', '88%'),
    ('2015/16', '89%'),
    ('2016/17', '90%'),
    ('2017/18', '90%'),
    ],
    'correct': '91%'
}

gov2 = {
    'metric_name': 'Number of stops and searches carried out rate per 1000 Black people in the UK', 
    'subject_time': '2020/21',
    'collecting_org': 'the Race Disparity Unit',
    'history': [
    ('2010/11', '31.2'),
    ('2016/17', '29'),
    ('2017/18', '29'),
    ('2018/19', '38'),
    ('2019/20', '54'),
    ],
    'correct': '52.6'
}

gov3 = {
    'metric_name': 'Number of non-fatal injuries to the workforce on the mainline of British Rail',
    'subject_time': 'Apr 2008 to Mar 2009',
    'collecting_org': 'the Office of Rail and Road',
    'history':[
    ('Apr 2003 to Mar 2004', '7,132'),
    ('Apr 2004 to Mar 2005', '7,154'),
    ('Apr 2005 to Mar 2006', '6,906'),
    ('Apr 2006 to Mar 2007', '6,490'),
    ('Apr 2007 to Mar 2008', '7,299'),
    ],
    'correct': '7,101'
}

gov4 = {
    'metric_name': 'Number of attributable deaths to PM2.5 concentration assuming 6% mortality coefficient',
    'subject_time': 'Chessington North and Hook',
    'collecting_org': 'the Greater London Authority',
    'history': [
    ('Stanley', '4'),
    ('Alexandra', '5'),
    ('Berrylands', '6'),
    ('Beverley', '6'),
    ('Canbury', '7'),
    ],
    'correct': '5'
}

gov5 = {
    'metric_name': 'Agricultural Price Index for All Agricultural Inputs',
    'subject_time': '01/03/2018',
    'collecting_org': 'the Department for Environment, Food & Rural Affairs',
    'history': [
    ('01/10/2017', '93.63083793127743'),
    ('01/11/2017', '94.81980286804415'),
    ('01/12/2017', '95.88209385405582'),
    ('01/01/2018', '95.31283262921369'),
    ('01/02/2018', '95.42932334834583'),
    ],
    'correct': '96.09044337785106'
}

if __name__ == '__main__':
    # test_5_shot(None, None, **control, template=template_a, template_prefix=None)
    # ask_information_retrieval(None, None, **control, template=llama_it_template)
    # ask_data_location(None, None, **control, template=llama_it_template)

    

    pass
