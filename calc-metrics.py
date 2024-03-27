from evaluate import load
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])

preds = df['predictions'].tolist()
sans = df['Sanskrit'].tolist()


print(len(preds), len(sans))
metric = load('sacrebleu')
bl = metric.compute(predictions=preds, references = sans)
print('BLEU is ', bl)

metric = load('chrf')
chrf = metric.compute(predictions=preds, references = sans)
print('ChrF is ', chrf)

