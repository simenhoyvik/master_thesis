import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from utils import calc_map, clean, create_dir_if_not_exists, get_avg_studies_pr_query, load_pickle, normalize, prepare, preprocess_df_bm25, remove_stop, replace_urls, save_pickle, tokenize

def process_doi(doi):
    if doi == None: return None
    processed = "https://doi.org/" + doi
    return processed

def bm25_documents_clean(documents):
    documents_list = []
    for i, document in enumerate(documents):
        title = None
        pubmed_id = None
        abstract = None
        try: title  = document.title
        except: pass
        try: pubmed_id = document.pubmed_id
        except: pass
        try: abstract = document.abstract
        except: pass
        document = {
            'title' : title,
            'pubmed_id' : pubmed_id,
            'study_abstract' : abstract,
            'rank' : i+1
        }
        documents_list.append(document)
    df_result = pd.DataFrame(documents_list)
    df_result['text'] = df_result['title'] + '. ' + df_result['study_abstract']
    df_result['text_tokenized'] = df_result['text'].apply(lambda x: x.split())
    return df_result

def get_top_similar(model, df, query, n):
    corpus = df['clean_abstract'].tolist()
    doc_scores = model.get_scores(query)
    df["bm25_score"] = doc_scores
    df = df.sort_values(by=['bm25_score'], ascending=False)
    top_n_raw = model.get_top_n(query, corpus, n=n)
    top_n_rows = df[df['clean_abstract'].isin(top_n_raw)]
    return top_n_rows


class Bm25():
    def __init__(self, verbose, pubmed_n, reduced_n):
        self.verbose = verbose
        self.pubmed_n = pubmed_n
        self.reduced_n = reduced_n
        self.history_path = f'history/bm25/basic'

    def train_and_evaluate_model(self, text_combination, train, val, test, cleaning_type):
        model_name = f'{text_combination}-cleaning-{cleaning_type}'
        if self.verbose: print(f"Training model: {model_name}")
        train = preprocess_df_bm25(train, text_combination, cleaning_type, 'text')
        tokenized_corpus_train = train['text'].tolist()
        train_model = BM25Okapi(tokenized_corpus_train)
        train_acc, train_map = self.evaluate_model(train, train_model)

        val = preprocess_df_bm25(val, text_combination, cleaning_type, 'text')
        tokenized_corpus_val = val['text'].tolist()
        val_model = BM25Okapi(tokenized_corpus_val)
        val_acc, val_map = self.evaluate_model(val, val_model)

        test = preprocess_df_bm25(test, text_combination, cleaning_type, 'text')
        tokenized_corpus_test = test['text'].tolist()
        test_model = BM25Okapi(tokenized_corpus_test)
        test_acc, test_map = self.evaluate_model(test, val_model)

        result_dict = {}
        result_dict['train_acc'] = train_acc
        result_dict['train_map'] = train_map
        result_dict['val_acc'] = val_acc
        result_dict['val_map'] = val_map
        result_dict['test_acc'] = test_acc
        result_dict['test_map'] = test_map
        save_pickle(self.history_path, result_dict)

    def evaluate_model(self, df, model):
        queries = df['title'].unique()
        avg_true_studies, avg_tot_studies = get_avg_studies_pr_query(df)
        new_dfs = []
        grouped = df.groupby('qid')
        for name, sub_df in grouped:
            query = df.iloc[0]['title']
            tokenized_query = query.split()
            scores = model.get_scores(tokenized_query)
            df["rank"] = scores
            sub_df = pd.merge(sub_df, df[['qid','docid','rank']], left_on=['docid','qid'], right_on=['docid','qid'], how='inner')
            sub_df = sub_df.sort_values(by=['rank'], ascending=False)
            sub_df = sub_df.reset_index(drop=True)
            if len(sub_df) < avg_true_studies: top_n = int(len(sub_df)/2)
            else: top_n = int(len(sub_df)/2)
            sub_df['pred'] = np.where(sub_df.index < top_n, 1, 0)
            new_dfs.append(sub_df)
        merged = pd.concat(new_dfs)
        merged.rename(columns = {'label':'true'}, inplace = True)
        correct_preds = len(merged[(merged['pred'] == 1) & (merged['true'] == 1)])
        accurracy = correct_preds / (len(merged))
        map_score = calc_map(merged)
        return accurracy, map_score
                

    def calc_map_single_search(self, df):
        trues = 0
        ap = 0
        map = 0
        for i, (_, row) in enumerate(df.iterrows()):
            if row['label'] == 1:
                trues += 1
                ap += (trues+1) / (i+1)               
        if trues != 0: ap = ap / (trues)
        map += ap
        return map

    def bm_25_rerank(self,df, query, count_query_df):
        tokenized_query = query.split()
        df['tokenized_corpus'] = df['study_abstract'].apply(lambda x: x.split())
        tokenized_corpus = df['tokenized_corpus'].tolist()
        model = BM25Okapi(tokenized_corpus)
        scores = model.get_scores(tokenized_query)
        df["bm25_score"] = scores
        df = df.sort_values(by=['bm25_score'], ascending=False)
        df = df.reset_index(drop=True)
        top_n_pred = int(len(df) / 2)
        df['pred_1'] = np.where(df.index < top_n_pred, 1, 0)
        map = self.calc_map_single_search(df)
        return map      

    def check_if_trained_model(self):
        try:
            model = load_pickle("models/bm25/model.pickle")
            return model
        except:
            print("Cant find model")
            return None

    def evaluate(self, df_test):
        unique_search_queries = df_test['title'].unique()
        tot_map = 0
        for search_query in unique_search_queries:
            query_df = df_test.loc[df_test['title'] == search_query]
            tot_map += self.bm_25_rerank(query_df, search_query, self.pubmed_n)
        tot_map = tot_map / len(unique_search_queries)
        if self.verbose: 
            print("Evaluating BM25 ranking method")
            print(f"MAP = {tot_map}")


        

    