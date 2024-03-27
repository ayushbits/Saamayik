import numpy as np
import pandas as pd
import re, os, string, typing, gc, json
import csv
from datasets import load_dataset, load_metric
import torch

lang_code = "san_Deva"

## Getting Test Data
# path = '../data/rawdata/English-'+language+'.csv'
# test_source, test_target = data_augmentation_2(path,language)

## The other way
# en_test_df = pd.read_csv("../data/ood-test/mkb/mkb.en", header=None, names =['English'])
# sa_test_df = pd.read_csv("../data/ood-test/mkb/mkb.sa", header=None,  names =['Sanskrit'])

en_test_df = pd.read_csv("../data/final_data/test.en", header=None, names =['English'])
sa_test_df = pd.read_csv("../data/final_data/test.sa", header=None,  names =['Sanskrit'])

test_df = pd.concat([en_test_df, sa_test_df], axis =1)
test_df = test_df.dropna()
test_source = test_df["English"].tolist()
test_target = test_df["Sanskrit"].tolist()


# modelling
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

print('\n\nLoading Model.......')
model_path = 'nllb-1.3B'
# model_path = 'facebook/nllb-200-1.3B'
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path =model_path) #path stored at the disk
tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-1.3B', do_lower_case=False, use_fast=False, keep_accents=True, src_lang="eng_Latn", tgt_lang="san_Deva", max_length = 500)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)
model= torch.nn.DataParallel(model).to(device)

metric_bleu = load_metric('sacrebleu')
metric_chrf = load_metric('chrf')
def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

def compute_metrics(eval_preds):
    #print("compute_met called")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, padding=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # print(labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, padding=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result_bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result_chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)
    result = {'bleu': result_bleu['score'], 'chrf' : result_chrf['score']}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result['gen_len'] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    #print("compute-met-end")
    return result

def compute_metrics_2(preds, labels):
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds, decoded_labels = postprocess_text(preds, labels)
    result_bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result_chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)
    result = {'bleu': result_bleu['score'], 'chrf' : result_chrf['score']}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result['gen_len'] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


print("\n\nInference Started.........")

batch_size = 100 # fix batch size depending on the gpu, this one works for 80GB
start = 0
all_predictions = []

while True:
    end = start + batch_size
    if end >= len(test_target):
        end = len(test_target)
    labels = [[x] for x in test_target[start:end]] # ground truth
    model_inputs = tokenizer(test_source[start:end], return_tensors="pt", padding = True, truncation = True)
    model_inputs.to(device)

    gen_tokens = model.module.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code])
    predictions = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True )
    predictions = [pred.strip() for pred in predictions]
    all_predictions.extend(predictions)
    print(f'Done with start: {start}, end: {end}')
    start = end
    if start >= len(test_target):
        break

language_code = {'English':'en', 'Sanskrit':'sa'}#, 'Kannada':'kn'}

s_lang = 'en'
t_lang = 'sa'
source_lang = 'English'
target_lang = 'Sanskrit'
predictions = all_predictions
print(len(test_source), len(test_target), len(predictions))
predictions_data = pd.DataFrame({f'{source_lang}':test_source, f'{target_lang}':test_target,'predictions':  predictions})
predictions_data.to_csv(f'nllb_{source_lang}_{target_lang}_predictions_test.csv')
result = compute_metrics_2(predictions, test_target)
print('Result is ', result)
