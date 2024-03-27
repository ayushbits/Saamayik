# Script to translate text using Google Translate API keys
import requests
import csv
import pandas as pd
import time
from tqdm import tqdm

api_key = "YOUR_API_KEY"

en_test_df = pd.read_csv("data/mkb/mkb.en", header=None, names =['English'])
sa_test_df = pd.read_csv("data/mkb/mkb.sa", header=None,  names =['Sanskrit'])

print('Testing Mann Ki Baat')
test_df = pd.concat([en_test_df, sa_test_df], axis =1)
test_df = test_df.dropna()


def translate(text, lang):
    r = requests.get(
    "https://translation.googleapis.com/language/translate/v2",
    params = {
        "key": api_key,
        "q": text,
        "target": lang,
        "alt":"json",
        "source":"en"
    }
    )
        
    translation = r.json()['data']['translations'][0]['translatedText']
    return translation

with open('Google_translate_mkb.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile)
    fields = ['English', 'predictions', 'Sanskrit'] 
    csvwriter.writerow(fields) 

    for index, row in tqdm(test_df.iterrows()):
            eng = row['English']
            sans = row['Sanskrit']
            preds = translate(eng, lang = 'sa')
            csvwriter.writerow([eng, sans, preds])
            time.sleep(1)
