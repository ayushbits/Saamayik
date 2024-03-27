import pandas as pd
import numpy as np
from datasets import Dataset
import transformers
from datasets import  DatasetDict

en_train_df = pd.read_csv("../data/final_data/train.en", header=None, names =['English'])
sa_train_df = pd.read_csv("../data/final_data/train.sa", header=None,  names =['Sanskrit'])

en_dev_df = pd.read_csv("../data/final_data/dev.en", header=None, names =['English'])
sa_dev_df = pd.read_csv("../data/final_data/dev.sa", header=None,  names =['Sanskrit'])

en_test_df = pd.read_csv("../data/final_data/test.en", header=None, names =['English'])
sa_test_df = pd.read_csv("../data/final_data/test.sa", header=None,  names =['Sanskrit'])

# en_test_df = pd.read_csv("../data/ood-test/mkb/mkb.en", header=None, names =['English'])
# sa_test_df = pd.read_csv("../data/ood-test/mkb/mkb.sa", header=None,  names =['Sanskrit'])

train_df = pd.concat([en_train_df, sa_train_df], axis =1)
train_df = train_df.dropna()
dev_df = pd.concat([en_dev_df, sa_dev_df], axis =1)
dev_df = dev_df.dropna()
test_df = pd.concat([en_test_df, sa_test_df], axis =1)
test_df = test_df.dropna()

dataset = DatasetDict()
dataset['train'] = Dataset.from_pandas(train_df, split ='train')
dataset['val'] = Dataset.from_pandas(dev_df, split ='val')
dataset['test'] = Dataset.from_pandas(test_df, split ='test')
print(dataset)


# dataset = dataset.train_test_split(test_size=0.1, shuffle=False)

# model_name = 'indicbart-final_data/checkpoint-39000'
model_name = 'ai4bharat/IndicBART'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
#

# source_lang = 'English'
# target_lang = 'Sanskrit'
# language_code = {'English':'en', 'Sanskrit':'sa'}
# s_lang = 'en'
# t_lang = 'sa'

source_lang = 'Sanskrit'
target_lang = 'English'
language_code = {'English':'en', 'Sanskrit':'sa'}
s_lang = 'sa'
t_lang = 'en'

max_input_length = 512
max_target_length = 512

def preprocess_function(examples):
        inputs = [example + ' </s>' + f' <2{s_lang}>' for example in examples[source_lang]]

        targets = [f'<2{t_lang}> ' + example + ' </s>' for example in examples[target_lang]]
        # print('targets --', targets)
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs

batch_size = 128
tokenized_datasets = dataset.map(preprocess_function, batched=True)
#NIos + MKB--12K approx sentence-- 56mins

# print(tokenized_datasets)
count = 0
for token in tokenized_datasets['test']['labels']:
    val = tokenizer.decode(token)
    # print(token)
    # print(val)
    if "[UNK]" in val or "<unk>" in val:
        count+=1


print(f"Number of sentence having Unknown tokens: {count}")
        

tokenized_datasets.save_to_disk("data-sa-en/indicbart_sa_en_final")

# reload_ = load_from_disk("data/indicbart_en_sa")

# print("Reload-- ",reload_)