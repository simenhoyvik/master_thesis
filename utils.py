from collections import defaultdict
import html
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from textacy import preprocessing
import textacy.preprocessing as tprep
import nltk
from nltk.stem import PorterStemmer
import spacy
from nltk.tokenize import wordpunct_tokenize
#nlp = spacy.load('en_core_web_sm')
import string
import gensim
import gensim.downloader as api
from os.path import exists
from random import randrange
import pandas as pd
import nlpaug.augmenter.word as naw
import torch.utils.data as data_utils
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer, BertConfig, AlbertConfig, RobertaConfig, AutoTokenizer, AutoConfig

from sys import platform
if platform == "linux": import pickle5 as pickle
else: import pickle

import nltk.corpus
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stopwords = set(stop_words)
stopwords.update(["nan"])

def df_to_list(df):
    res = []
    for column in df.columns:
        li = df[column].tolist()
        res.append(li)
    return res

def to_url(input):
    url = "https://pubmed.ncbi.nlm.nih.gov/" + input[0:8] + "/"
    return url

def get_stopwords():
    return set(nltk.corpus.stopwords.words('english'))

def save_pickle(filename, object):
    with open(filename, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        loaded_object = pickle.load(handle)
    return loaded_object

def load_pickles(filenames):
    for filename in filenames:
        pass

def create_dir_if_not_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except:
        pass

def preprocess_df_bm25(df, text_combination, cleaning_type, text_column):
    df[text_column] = ''
    columns = text_combination.split()
    if len(columns) == 1:
        column = columns[0]
        if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean)
        else: df[text_column] = df[text_column] + df[column]
    else:
        for i, column in enumerate(columns): 
            if i == len(columns)-1:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean)
                else: df[text_column] = df[text_column] + df[column]
            else:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean) + ". "
                else: df[text_column] = df[text_column] + df[column] + ". "
    df[text_column] = df[text_column].apply(lambda x: x.split())
    return df


def clean_old(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text) 
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_2(text): # Cleaning without removing numbers
    # Lowercase the text
    text = text.lower()
    # Remove line breaks
    text = re.sub(r'\n', '', text)
    # Remove links
    text = re.sub(r'https*\S+', ' ', text)
    # Remove stopwords
    text = text.split()
    text = [word for word in text if word not in stopwords]
    # Lemmatize
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(y) for y in text]
    text = ' '.join(text)    

def load_word2vec_model():
    #model = api.load('word2vec-google-news-300')
    model = api.load('glove-wiki-gigaword-50')
    return model

def clean(text): # Complete cleaning
    # Lowercase the text
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z]', ' ', text)
    # Remove line breaks
    text = re.sub(r'\n', '', text)
    # Remove links
    text = re.sub(r'https*\S+', ' ', text)
    # Remove stopwords
    text = text.split()
    text = [word for word in text if word not in stopwords]
    # Remove numbers
    text = [re.sub(r'\w*\d\w*', '', w) for w in text]
    # Lemmatize
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(y) for y in text]
    text = ' '.join(text)
    '''
    text = text.lower() # lowercase everything
    text = text.encode('ascii', 'ignore').decode()  # remove unicode characters
    text = re.sub(r'https*\S+', ' ', text) # remove links
    text = re.sub(r'http*\S+', ' ', text) # remove links
    text = re.sub(r'\'\w+', '', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s[^\w\s]\s', '', text)
    
    text = ' '.join([word for word in text.split(' ') if word not in stopwords])
    text = re.sub(r'@\S', '', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # remove single letters and numbers surrounded by space
    #text = re.sub(r'\s[a-z]\s|\s[0-9]\s', ' ', text)
    return text
    
    stem="Spacy"
    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    #elif stem == 'Spacy':
    #    text_filtered = nlp(' '.join(text_filtered))
    #    text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)
    '''
    return text

def replace_urls(text):
    return preprocessing.replace.urls(text)

def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text

def tokenize(text):
    return wordpunct_tokenize(text)

def remove_stop(tokens):
    return [t for t in tokens if t.lower() not in get_stopwords()]

def remove_punct(text):
    return text.translate(str.maketrans('','', string.punctuation))

def calc_map(df):
    grouped = df.sort_values(['qid','pred'],ascending=False).groupby('qid')
    map = 0
    for name, group in grouped:
        trues = 0
        ap = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['true'] == 1:
                trues += 1
                ap += (trues) / (i+1)
        if trues != 0:
            ap = ap / (trues)
            map += ap
    map = map / len(grouped)
    return map

def prepare(text, pipeline):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens

def clean_df(df, column):
    df[column] = df[column].progress_map(clean)
    return df

def extract_tfidf_features(df, model):
    df['label_0'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    df['label_1'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
    y = torch.tensor(df[['label_0', 'label_1']].values.astype(np.float32))
    df = clean_df(df)
    tfidf = model.transform(df["clean_text"])
    tfidf_dense = tfidf.todense()
    X = torch.Tensor(tfidf_dense)
    data = data_utils.TensorDataset(X,y)
    return data

def dataset_to_dataloader(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, val_loader, test_loader

def get_avg_studies_pr_query(df):
    qids = df['qid'].unique()
    avg_pos = 0
    avg_neg = 0
    for qid in qids:
        q_df = df.loc[df['qid'] == qid]
        avg_pos += len(q_df.loc[q_df['label'] == 1])
        avg_neg += len(q_df.loc[q_df['label'] == 0])
    avg_pos = int(avg_pos / len(qids))
    avg_neg = int(avg_neg / len(qids))
    avg_tot = avg_pos + avg_neg
    return avg_pos, avg_tot

def preprocess_df_1(df, verbose, text_combination, cleaning):
    df["text"] = ""
    columns = text_combination.split()
    for i, column in enumerate(columns):
        if i == (len(columns)-1): df["text"] =  df["text"] + df[column] 
        else: df["text"] = df["text"] + df[column] + ". "
    df = df[["text", "label", "qid"]]
    if cleaning: df['text'] = df['text'].apply(clean)
    if verbose: print("Finished preparing data for approach 1")
    return df

def split_groupby(df, split_ratio, groupby_id, reduce, verbose):
    if reduce != None:
        splitter = GroupShuffleSplit(test_size=reduce, n_splits=2, random_state = 7)
        split = splitter.split(df, groups=df[groupby_id])
        inds, _ = next(split)
        df = df.iloc[inds]
        
    splitter = GroupShuffleSplit(test_size=split_ratio, n_splits=2, random_state = 7)
    split = splitter.split(df, groups=df[groupby_id])
    train_inds, val_test_inds = next(split)

    train = df.iloc[train_inds]
    val_test = df.iloc[val_test_inds]

    splitter = GroupShuffleSplit(test_size=0.5, n_splits=2, random_state = 7)
    split = splitter.split(val_test, groups=val_test[groupby_id])
    val_inds, test_inds = next(split)

    val = val_test.iloc[val_inds]
    test = val_test.iloc[test_inds]

    train = train.groupby(groupby_id).apply(lambda x: x.sample(frac=1))
    val = val.groupby(groupby_id).apply(lambda x: x.sample(frac=1))
    test = test.groupby(groupby_id).apply(lambda x: x.sample(frac=1))

    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return train, val, test

def init_qid_dict(prev_dict, lst):
    unique_set = set(lst)
    unique_list = list(unique_set)
    for qid in unique_list:
        if qid not in prev_dict:
            prev_dict[qid] = []
    return prev_dict

def all_items_same(lst):
    lst = lst.tolist()
    if([lst[0]]*len(lst) == lst):
        return True
    else:
        return False

def find_split_in_list(lst):
    prev = None
    for i, value in enumerate(lst):
        if prev == None:
            prev = value
            continue
        if value != prev:
            all_same = all_items_same(lst[i:])
            return i, all_same

def get_model_type(pre_trained_model_name):
    if pre_trained_model_name == 'bert-base-cased' or pre_trained_model_name =='bert-base-uncased': return 'bert'
    elif pre_trained_model_name == 'albert-base-v2': return 'albert'
    elif pre_trained_model_name == 'roberta-base': return 'roberta'
    else: return None

def get_tokenizer(model_type, pre_trained_model_name):
    '''
    tokenizer = None
    if model_type == 'bert': tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
    elif model_type == 'albert': tokenizer = AlbertTokenizer.from_pretrained(pre_trained_model_name)
    elif model_type == 'roberta': tokenizer = RobertaTokenizer.from_pretrained(pre_trained_model_name)
    else: tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)'''
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    return tokenizer

def get_model_config(model_type, pre_trained_model_name):
    model_config = None
    if model_type == 'bert': model_config = BertConfig.from_pretrained(pre_trained_model_name, output_hidden_states=True)
    elif model_type == 'albert': model_config = AlbertConfig.from_pretrained(pre_trained_model_name, output_hidden_states=True)
    elif model_type == 'roberta': model_config = RobertaConfig.from_pretrained(pre_trained_model_name, output_hidden_states=True)
    else: model_config = AutoConfig.from_pretrained(pre_trained_model_name, output_hidden_states=True)
    return model_config
    


def remove_too_large_abstracts(df, min_max_amount, text_column):
    df['text_length'] = df[text_column].apply(lambda x: (len(x.split())))
    df = df.loc[df['text_length'] > min_max_amount[0]]
    df = df.loc[df['text_length'] < min_max_amount[1]]
    df = df.drop(columns=['text_length'])
    return df



def preprocess_df_bert_2_4(df, text_combination, cleaning_type, max_abstract_length, text_column):
    df[text_column] = ''
    columns = text_combination.split()
    if len(columns) > 2:
        for i, column in enumerate(columns): 
            if i == len(columns)-1:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean)
                else: df[text_column] = df[text_column] + df[column]
            elif i == 0:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean) + " [SEP] "
                else: df[text_column] = df[text_column] + df[column] + " [SEP] "
            else:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean) + ". "
                else: df[text_column] = df[text_column] + df[column] + ". "
    else:
        for i, column in enumerate(columns): 
            if i == len(columns)-1:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean)
                else: df[text_column] = df[text_column] + df[column]
            else:
                if cleaning_type: df[text_column] = df[text_column] + df[column].apply(clean) + " [SEP] "
                else: df[text_column] = df[text_column] + df[column] + " [SEP] "
    if max_abstract_length != None: df = remove_too_large_abstracts(df, max_abstract_length, text_column)
    return df

def average_scores(history, n_epochs, cv):
    avgDict = defaultdict(list)
    train_acc_avg = 0
    train_loss_avg = 0
    train_map_avg = 0
    val_acc_avg = 0
    val_loss_avg = 0
    val_map_avg = 0
    for i in range(n_epochs):
        for j in range(cv):
            train_acc_avg += history[j]['train_acc'][i]
            train_loss_avg += history[j]['train_loss'][i]
            train_map_avg += history[j]['train_map'][i]
            val_acc_avg += history[j]['val_acc'][i]
            val_loss_avg += history[j]['val_loss'][i]
            val_map_avg += history[j]['val_map'][i]
        avgDict['train_acc'].append(train_acc_avg/cv)
        avgDict['train_loss'].append(train_acc_avg/cv)
        avgDict['train_map'].append(train_acc_avg/cv)
        avgDict['val_acc'].append(train_acc_avg/cv)
        avgDict['val_loss'].append(train_acc_avg/cv)
        avgDict['val_map'].append(train_acc_avg/cv)
    return avgDict

def preprocess_df_pairwise(df, text_combination, cleaning_type, max_abstract_length):
    text_combination_1 = text_combination[0]
    text_combination_2 = text_combination[1]
    df_pos = preprocess_df_bert_2_4(df, text_combination_1, cleaning_type, max_abstract_length, "text_pos")
    df_neg = preprocess_df_bert_2_4(df, text_combination_2, cleaning_type, max_abstract_length, "text_neg")
    df_pos['text_neg'] = df_neg['text_neg']
    return df_pos

def preprocess_df_bert_3(df, text_columns, cleaning_type, max_abstract_length):
    for i, column in enumerate(text_columns): 
        if cleaning_type: df[column] = df[column].apply(clean)
        if max_abstract_length != None: df = remove_too_large_abstracts(df, max_abstract_length, column)
    return df

def preprocess_df_bert_6(df, text_combination, cleaning_type, max_abstract_length, sub_approach):
    if sub_approach == 1:
        df = preprocess_df_bert_3(df, text_combination[sub_approach], cleaning_type, max_abstract_length)
    else:
        df = preprocess_df_bert_2_4(df, text_combination[sub_approach][0], cleaning_type, max_abstract_length, 'text_1')
        df = preprocess_df_bert_2_4(df, text_combination[sub_approach][1], cleaning_type, max_abstract_length, 'text_2')
    return df

def feature_scaling(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X

def print_data_size(X_train, X_test):
    print('Size of Training Data ', X_train.shape[0])
    print('Size of Test Data ', X_test.shape[0])   

def sample_random_query(df, column, verbose):
    index = randrange(len(df))
    example_query = df.iloc[index][column]
    if verbose:
        print("Query: \n", example_query)
    return example_query

def check_if_model_exists(path):
    if exists(path):
        model = load_pickle(path)
        return model
    else:
        return None

def precision_at_k(df, k):
    df = df[:k+1]
    df_corrects = df[(df.pred == 1) & (df.label == 1)]
    return len(df_corrects) / len(df)

def rel_at_k(df, k):
    value = df.iloc[k]['label']
    return value

def get_gtp(df, qid):
    sub_df = df[df.qid == qid]
    trues = sub_df[sub_df.label == 1]
    return trues.shape[0]

def get_unique_queries(df):
    return df['qid'].unique()

def get_df_by_id(df, qid):
    df = df.loc[df['qid'] == qid]
    return df

def average_precision(model, df, qid, n):
    sum = 0
    sub_df = get_df_by_id(df, qid)
    gtp = get_gtp(sub_df, qid)

    prediction = model.predict_proba(sub_df["text"])
    prediction = pd.DataFrame(prediction, columns = ['pred_0', 'pred_1'])
    prediction['pred'] = np.where(prediction['pred_1'] >= prediction['pred_0'], 1, 0)
    prediction = prediction.sort_index(ascending=True)
    prediction = prediction.reset_index()
    sub_df = sub_df.reset_index()
    sub_df = pd.concat([prediction, sub_df], axis=1)
    sub_df = sub_df.sort_values(by=['pred_1'], ascending=False)

    if len(sub_df) < n: n = len(sub_df)

    for k in range(0, n):
        p_k = precision_at_k(sub_df, k)
        rel_k = rel_at_k(sub_df, k)
        sum += (p_k * rel_k)
    return sum / n

def mean_average_precision_at_n(model, df, n):
    unique_queries = get_unique_queries(df)
    Q = len(unique_queries)
    sum = 0
    for i, qid in enumerate(unique_queries):
        sum += average_precision(model, df, qid, n)
    map = sum / Q
    print(f"Mean Average Precision, with N = {n}, is {map}")