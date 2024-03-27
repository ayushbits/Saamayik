import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import os

def read_bible(path='data/bible/aligned-bible.csv'):
    bible_csv = pd.read_csv(path,  names=['English','Hindi','Sanskrit'], header=None)
    return bible_csv

def read_gita(path='data/gitasopanam/'):
    eng, sa = [], []
    with open(os.path.join(path, 'english.txt')) as f:
        for line in f.readlines():
            eng.append(line.strip())
    with open(os.path.join(path, 'sanskrit.txt')) as f:
        for line in f.readlines():
            sa.append(line.strip())
    df = pd.DataFrame({'English': eng, 'Sanskrit':sa}, columns =['English','Sanskrit'])
    return df


def read_mkgandhi(path='data/mkgandhi/'):
    full_df = pd.DataFrame()
    for name in glob(path + '*.xlsx'):
        df_dict = pd.read_excel(name, sheet_name=None, usecols ='A, B, C', header=None)
        full_df = pd.concat(df_dict.values(), ignore_index=True, names =['English', 'Hindi', 'Sanskrit'])

    full_df.rename(columns={0:'English',1: 'Hindi',2:'Sanskrit'}, inplace= True)
    return full_df
     
     
def read_nios(path='data/nios/'):
    
    full_df = pd.DataFrame()
    for name in glob(path + '*.xlsx'):
        dfname =  pd.ExcelFile(name)
        for sheet in dfname.sheet_names:
            df_dict = pd.read_excel(name, sheet_name=sheet, usecols ='A, B, C', header=None)
            full_df = pd.concat([full_df, df_dict], ignore_index= True, names =['English', 'Hindi', 'Sanskrit'])
    full_df.rename(columns={0:'English',1: 'Hindi',2:'Sanskrit'}, inplace= True)
    return full_df

def read_spoken(path='data/spoken-tutorials/'):
    for name in glob(path +'*.csv'):
        # print(name)
        if 'English' in name:
            eng  = pd.read_fwf(name, names =['English'], header = None)
        if 'Hindi' in name:
            hin  = pd.read_csv(name, names=['Hindi'], header = None)
        if 'Sanskrit' in name:
            # sanskrit  = pd.read_fwf(name, names =['Sanskrit'], header = None)
            sanskrit = []
            with open(name, 'r') as f:
                for s in f.readlines():
                    sanskrit.append(s.strip())
            sanskrit_df = pd.DataFrame(sanskrit, columns= ['Sanskrit'])
    # hin.rename(columns='Hindi')
    eng1 = eng.join(hin)
    eng2 = eng1.join(sanskrit_df)
    # final_df = pd.concat([eng,hindi,sanskrit_df], axis=1, ignore_index= True).reset_index(drop=True)
    return eng2
     
     
def read_mkb(path='data/mkb/'):
    
    full_df = pd.DataFrame()
    for name in glob(path + '*.xlsx'):
        if 'Amrith' in name:
            dfname =  pd.ExcelFile(name)
            df_dict = pd.read_excel(name, sheet_name=None, usecols ='A, B, D', header=None)
            # pd.concat([full_df, df], ignore_index= True)

            full_df = pd.concat(df_dict.values(), ignore_index=True, names =['English', 'Hindi', 'Sanskrit'])
            # print(name, len(full_df.columns))
        else:
            df = pd.read_excel(name, usecols ='A, B, D', header=None)
            # print(name, len(full_df.columns))
            full_df = pd.concat([full_df, df], ignore_index= True, names =['English', 'Hindi', 'Sanskrit'])
    full_df.rename(columns={0:'English',1: 'Hindi',3:'Sanskrit'}, inplace= True)
    return full_df

def split_train_test(df):
    train, test = train_test_split(df, test_size=0.1, random_state= 7, shuffle= True)
    test, dev = train_test_split(test, test_size=0.5, random_state= 7, shuffle= False)
    
    return train, dev, test

def write_data(df, split='train', path='data/final_data/'):
    for column in df:
        if 'hindi' in column.lower():
            df[column].to_csv(path + split+'.hi', index = False, header = False)
        if 'english' in column.lower():
            df[column].to_csv(path + split+'.en', index = False, header = False, encoding ='utf-8')
        if 'sanskrit' in column.lower():
            df[column].to_csv(path + split+'.sa', index = False, header = False, encoding ='utf-8')


if __name__ == "__main__":

    df_bible = read_bible()
    df_nios = read_nios()
    df_spoken = read_spoken()
    df_gita = read_gita()
    # df_mkg = read_mkgandhi()
    # df_mkb = read_mkb()

    
    
    df = pd.concat([df_bible, df_spoken, df_gita, df_nios], ignore_index= True, names =['English', 'Hindi', 'Sanskrit'])
    # df = pd.concat([df_bible, df_spoken ], ignore_index= True, names =['English', 'Hindi', 'Sanskrit'])
    
    # print('Final merged df columns ', len(df.columns))
    # print('Final merged df_mkb columns ', len(df_mkb.columns))
    # print('Final merged df_spoken columns ', len(df_spoken.columns))
    # print('Final merged df_bible columns ', len(df_bible.columns))
    print('len of df hindi ', len(df['Hindi']))
    print('len of df English ', len(df['English']))
    print('len of df Sanskrit ', len(df['Sanskrit']))

    # df['English'] = df['English'].str.replace('^"+|"+$|"""', '')
    # df['Hindi'] = df['Hindi'].str.replace('^"+|"+$|"""', '')
    # df['Sanskrit'] = df['Sanskrit'].str.replace('^"+|"+$|"""', '')

    # df_mkg['English'] = df_mkg['English'].str.replace('^"$"', '')
    # df_mkg['Hindi'] = df_mkg['Hindi'].str.replace('^"+|"+$|"""', '')
    # df_mkg['Sanskrit'] = df_mkg['Sanskrit'].str.replace('^"+|"+$|"""', '')

    df['English'] = df['English'].str.strip()
    df['Hindi'] = df['Hindi'].str.strip()
    df['Sanskrit'] = df['Sanskrit'].str.strip()
    df['English'] = df['English'].str.replace(r'\n', '')
    df['Hindi'] = df['Hindi'].str.replace(r'\n', '')
    df['Sanskrit'] = df['Sanskrit'].str.replace(r'\n', '')

    df['English'] = df['English'].str.strip()
    df['Hindi'] = df['Hindi'].str.strip()
    df['Sanskrit'] = df['Sanskrit'].str.strip()
    # df_mkg['English'] = df_mkg['English'].str.replace(r'\n', '')
    # df_mkg['Hindi'] = df_mkg['Hindi'].str.replace(r'\n', '')
    # df_mkg['Sanskrit'] = df_mkg['Sanskrit'].str.replace(r'\n', '')
    
    df.dropna()
    # df_mkg.dropna()
    print('len of df hindi ', len(df['Hindi']))
    print('len of df English ', len(df['English']))
    print('len of df Sanskrit ', len(df['Sanskrit']))

    train, test, dev = split_train_test(df)
    write_data(train)
    write_data(test, split='test')
    write_data(dev, split='dev')

    # write_data(df, split='all1')
    # write_data(df_mkg, split='mkg')


    # Run below commands to copy to respective folders
