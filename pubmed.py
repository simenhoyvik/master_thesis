import numpy as np
import pandas as pd
from pymed import PubMed

from utils import calc_map, get_avg_studies_pr_query, save_pickle

def search_pubmed(query, max_result):
    pubmed = PubMed(tool="MyTool", email="qug020@uib.no")
    try:
        documents = pubmed.query(query, max_results=max_result)
        documents = list(documents)
        return documents
    except:
        documents = None
        return None
    
def search_and_process_pubmed(query, max_result):
    documents = {}
    results = search_pubmed(query, max_result)

    if results == None:
        return None

    for i, result in enumerate(results):
        document = {}
        document['title'] = query
        document['index'] = i
        try: document['study_title'] = result.title
        except: document['study_title'] = None
        try: document['study_abstract'] = result.abstract
        except: document['study_abstract'] = None
        try: document['pubmed_id'] = result.pubmed_id
        except: document['pubmed_id'] = None
        document['label'] = 0
        document['qid'] = result.pubmed_id
        documents[i] = document

    test = pd.DataFrame.from_dict(documents, orient='index',
                       columns=['index', 'title', 'study_title', 'study_abstract', 'pubmed_id', 'label', 'qid'])
    return test

class PubMedModel:
    def __init__(self, verbose, force_restart):
        self.verbose = verbose
        self.force_restart = force_restart
        self.history_path = f'history/pubmed/basic'

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

    def evaluate(self, pubmed_search_amount, df_test):
        dfs = []
        unique_search_queries = df_test['title'].unique()
        tot_map = 0
        for search_query in unique_search_queries:
            query_df = df_test.loc[df_test['title'] == search_query]
            date = query_df['date'].to_numpy()[0]
            qid = query_df['qid'].to_numpy()[0]
            date = pd.to_datetime(date).date()
            documents = search_pubmed(search_query, pubmed_search_amount)
            if documents == None: continue
            top_n_true = int(len(documents) / 2)
            documents_list = []
            ranker = 0
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
                try:
                    study_date = document.publication_date
                    if type(study_date) == str:
                        study_date = pd.to_datetime(study_date).date()
                except: pass
                if study_date <= date:
                    document = {
                        'qid': qid,
                        'title' : search_query,
                        'pubmed_id' : pubmed_id,
                        'study_title' : title,
                        'study_abstract' : abstract,
                        'study_date': study_date,
                        'rank' : ranker+1
                    }
                    ranker += 1
                    documents_list.append(document)
            if len(documents_list) == 0:
                continue
            df_result = pd.DataFrame(documents_list)
            df_result['rank'] = df_result['rank'].astype(int)
            df_result['pred'] = np.where(df_result['rank'] <= int(len(df_result)/2), 1, 0)
            df_result = df_result.join(query_df.set_index('pubmed_id')[['label']], on='pubmed_id')
            query_df['rank'] = None
            query_df['pred'] = 0
            df_result = pd.concat([df_result, query_df[['qid','title', 'pubmed_id', 'study_title', 'study_abstract', 'rank', 'pred', 'label']]])
            df_result = df_result.drop_duplicates('pubmed_id', keep='first')
            df_result = df_result.reset_index()
            df_result = df_result.drop('index', axis=1)
            df_result = df_result[df_result['rank'].notna()]
            df_result['label'] = np.where(df_result['label'].isnull(), 0, df_result['label'])
            dfs.append(df_result)
            #tot_map += self.calc_map_single_search(df_result)
        #tot_map = tot_map / len(unique_search_queries)
        merged = pd.concat(dfs)
        merged.rename(columns = {'label':'true'}, inplace = True)
        correct_preds = len(merged[(merged['pred'] == 1) & (merged['true'] == 1)])
        accurracy = correct_preds / (len(merged))
        map_score = calc_map(merged)
        result_dict = {}
        result_dict['acc'] = accurracy
        result_dict['map'] = map_score
        result_dict['pubmed_max_result'] = pubmed_search_amount
        save_pickle(self.history_path, result_dict)