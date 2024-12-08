from leakage_utils import *
from tqdm import tqdm
import pandas as pd

inf_models = [
    'meta-llama/Meta-Llama-3.1-8B',
    'google/gemma-2-2b',
    'Qwen/Qwen2.5-3B'
    ]

instr_models = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'google/gemma-2-2b-it',
    'Qwen/Qwen2.5-3B-Instruct'
]


data_names = ['bank_rate', 'smoke_alarms', 'stop_and_search',
              'rail_inj', 'air_poll', 'agr_price']
data = [control, gov1, gov2, gov3, gov4, gov5]

templates = [template_a, template_b, template_c]#, template_d]

inf_results = {}

for model_name in tqdm(inf_models):
    model, tokenizer = model_setup(model_name)
    inf_m_results = []
    for index in tqdm(range(len(data_names))):
        for template in templates:
            inf_m_results.append(test_0_shot(model, tokenizer, **data[index], template=template))
            inf_m_results.append(test_1_shot(model, tokenizer, **data[index], template=template))
            inf_m_results.append(test_5_shot(model, tokenizer, **data[index], template=template))

        inf_m_results.append(test_0_shot(model, tokenizer, **data[index], template=template_d, template_prefix=template_d_prefix))
        inf_m_results.append(test_1_shot(model, tokenizer, **data[index], template=template_d, template_prefix=template_d_prefix))
        inf_m_results.append(test_5_shot(model, tokenizer, **data[index], template=template_d, template_prefix=template_d_prefix))
    
    inf_results[model_name] = inf_m_results
    del(model)
    del(tokenizer)

instr_results = {}
for model_name in tqdm(instr_models):
    model, tokenizer = model_setup(model_name)
    instr_m_results = []
    for index in tqdm(range(len(data_names))):
        instr_m_results.append(ask_information_retrieval(model, tokenizer, **data[index], template=m_to_t[model_name]))
        instr_m_results.append(ask_data_location(model, tokenizer, **data[index], template=m_to_t[model_name]))

    instr_results[model_name] = instr_m_results
    del(model)
    del(tokenizer)
    
        
df = pd.DataFrame(inf_results)
df2 = pd.DataFrame(instr_results)
print(df)
print(df2)
# df.to_csv(...)



