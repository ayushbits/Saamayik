import pandas as pd
import numpy as np
import transformers
from transformers import Seq2SeqTrainer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer

from datasets import load_from_disk,load_metric

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)


# model_name = "ai4bharat/IndicBART"

# model_path = 'mbart-en-sa-final/checkpoint-93000'
# model_path = 'model-en-sa/mbart-itihasa/checkpoint-155000'
#model_path = 'model-en-sa/mbart-itihasa-ours/checkpoint-250000'
model_path = 'model-sa-en/mbart-sa-en-final/checkpoint-141000'
model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path =model_path) #path stored at the disk
tokenizer = MBart50TokenizerFast.from_pretrained(model_path, do_lower_case=False, use_fast=False, keep_accents=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  

batch_size = 16
tokenized_datasets = load_from_disk("data/mbart_sa_en_mkb")
#tokenized_datasets = load_from_disk("data/itihasa_en_sa")

args = Seq2SeqTrainingArguments(
            output_dir='en-sa-translation',
            evaluation_strategy='epoch',
            learning_rate=0.001,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.0001,
            save_total_limit=2,
            num_train_epochs=20,
            predict_with_generate=True,
            remove_unused_columns=True,
            label_smoothing_factor= 0.1)

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

trainer = Seq2SeqTrainer(
              model,
              args,
              train_dataset=tokenized_datasets['test'],
              eval_dataset=tokenized_datasets['test'],
              data_collator=data_collator,
              tokenizer=tokenizer,
              compute_metrics=compute_metrics,
              )  
for cb in trainer.callback_handler.callbacks:
    if isinstance(cb, transformers.integrations.NeptuneCallback):
        trainer.callback_handler.remove_callback(cb)
# trainer.train()
# trainer.evaluate()

predict_dataset = tokenized_datasets['test']
predict_results = trainer.predict(predict_dataset,
                                    metric_key_prefix='test',
                                    max_length = 200)

preds = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)
print(f"The prediction metric:{predict_results}")

predictions = tokenizer.batch_decode(preds,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True, )



predictions = [pred.strip() for pred in predictions]

language_code = {'English':'en', 'Sanskrit':'sa'}#, 'Kannada':'kn'}

s_lang = 'en'
t_lang = 'sa'

source_lang = 'English'
target_lang = 'Sanskrit'

predictions_data = pd.DataFrame({f'{source_lang}':tokenized_datasets['test'][source_lang.capitalize()], 
                                    f'{target_lang}':tokenized_datasets['test'][target_lang.capitalize()],
                                    'predictions':  predictions})

predictions_data.to_csv(f'mkb_{source_lang}_{target_lang}.csv')
