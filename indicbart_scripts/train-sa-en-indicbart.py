import pandas as pd
import numpy as np
from transformers import Seq2SeqTrainer
from transformers import MBartForConditionalGeneration, MBartConfig

from datasets import load_from_disk,load_metric
from transformers import EarlyStoppingCallback

from transformers import (AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

configuration = MBartConfig()
model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path ='indicbart')# stored at the disk

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  

batch_size = 16

tokenized_datasets = load_from_disk("data-sa-en/indicbart_sa_en_final")

args = Seq2SeqTrainingArguments(
            output_dir='model-sa-en/indicbart-sa-en-final_data',
            logging_dir= 'model-sa-en/indicbart-sa-en-final_data',
            learning_rate=0.0003,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.00001,
            save_total_limit=2,
            num_train_epochs=50,
            predict_with_generate=True,
            remove_unused_columns=True,
            label_smoothing_factor= 0.1,
            save_strategy = 'steps',
            warmup_steps=4000, 
            generation_max_length=200,
            evaluation_strategy='steps',
            eval_steps=3000,
            save_steps=3000,
            load_best_model_at_end=True,
            )
early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)

metric = load_metric('chrf')

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        # print(preds, labels)
        # print('inside posprocess text')
        # print(type(preds))
        # print('labels ', type(labels))
        return preds, labels

def compute_metrics(eval_preds):
        #print("compute_met called")
        preds, labels = eval_preds
        # print('preds ', preds[0])
        if isinstance(preds, tuple):
            preds = preds[0]
        
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # print(type(decoded_preds), len(decoded_preds))
        # print('labels ', type(labels), labels.shape)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # print(decoded_labels[1:10])
        # print(decoded_preds[1:10])
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        # pd.DataFrame({'Preds': decoded_preds, 'Labels':decoded_labels}).to_csv(f'eval_preds.csv')
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {'chrf': result['score']}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result['gen_len'] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        #print("compute-met-end")
        return result

trainer = Seq2SeqTrainer(
              model,
              args,
              train_dataset=tokenized_datasets['train'],
              eval_dataset=tokenized_datasets['val'],
              data_collator=data_collator,
              tokenizer=tokenizer,
              compute_metrics=compute_metrics,
              callbacks=[early_stopping],
              )  

trainer.train()
trainer.evaluate()

predict_dataset = tokenized_datasets['test']
predict_results = trainer.predict(predict_dataset,
                                    metric_key_prefix='predict')

predictions = tokenizer.batch_decode(predict_results.predictions,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)

predictions = [pred.strip() for pred in predictions]

language_code = {'English':'en', 'Sanskrit':'sa'}#, 'Kannada':'kn'}

# s_lang = 'en'
# t_lang = 'sa'

s_lang = 'sa'
t_lang = 'en'
source_lang = 'Sanskrit'
target_lang = 'English'

# source_lang = 'English'
# target_lang = 'Sanskrit'

predictions_data = pd.DataFrame({f'{source_lang}':tokenized_datasets['test'][source_lang.capitalize()], 
                                    f'{target_lang}':tokenized_datasets['test'][target_lang.capitalize()],
                                    'predictions':  predictions})

predictions_data.to_csv(f'IndicBart_test_{source_lang}_{target_lang}_predictions.csv')