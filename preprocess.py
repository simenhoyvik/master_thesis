import json
import os
import random
import numpy as np
import xmltodict
from pymed import PubMed
from os import walk
from difflib import SequenceMatcher
import xml.etree.ElementTree as et
import pandas as pd
import os
import math
from pymed import PubMed
from pubmed import search_pubmed
import string
import re

from utils import load_pickle, save_pickle

dir_path = './data/reviews_part_2'
reviews_filename = "./data/processed/all_reviews_part_2.pickle"
reviews_pandas_filename = "./data/processed/df_part_2.pickle"

title_sim_ratio = 0.98
title_sim_ratio_2 = 0.85
author_sim_ratio = 0.75
pubmed_search_amount = 2
minimum_amount_related_studies = 0

def count_files_in_dir(path):
    return len([name for name in os.listdir(path)])
 
def xml_to_dict(filename):
    try:
        with open(filename, encoding='unicode_escape') as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
        return data_dict
    except Exception as e: 
        print("Failed on xml to dict")
        print(e)
        return None

def few_study_amount(review):
    try:
        included_studies = review["COCHRANE_REVIEW"]["STUDIES_AND_REFERENCES"]["STUDIES"]["INCLUDED_STUDIES"]["STUDY"]
        excluded_studies = review["COCHRANE_REVIEW"]["STUDIES_AND_REFERENCES"]["STUDIES"]["EXCLUDED_STUDIES"]["STUDY"]
        if type(included_studies) is not list: return False
        if len(included_studies) < minimum_amount_related_studies: return True
        if type(excluded_studies) is not list: return False
        if len(excluded_studies) < minimum_amount_related_studies: return True
    except Exception as e:
        #print("FEW STUDY AMOUNT")
        #print(e)
        return True
    return False

def preprocess_all_xml_documents(path, n, reviews_filename):
    print("Amount of reviews: ", count_files_in_dir(path))
    reviews = {}
    for i, filename in enumerate(os.listdir(path)):
        if n != None:
            if i > n: break
        fullname = os.path.join(path, filename)
        xml_review = xml_to_dict(fullname)
        if xml_review == None: continue
        #if few_study_amount(xml_review): continue
        print("\n=================================================\n")
        print(f"Preprocessing file: {i}_{filename}")
        result = retrieve_useful_data_single(xml_review)
        if result != None: reviews[f"{i}_{filename}"] = result
        print(f"\n=============Finished retrieving number {i}=============\n")
    save_pickle(reviews_filename, reviews)

def get_subset_of_dict(origin_dict):
    keys = ('AU', 'TI', 'SO', 'YR', 'PB')
    return dict([(x,origin_dict[x]) for x in keys if x in origin_dict])

def convert_time_object(date):
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)

def retrieve_footnotes(footnotes):
    all_footnotes = ""
    if type(footnotes) is dict:
        if type(footnotes["P"]) is str:
            all_footnotes += footnotes["P"]
        elif type(footnotes["P"]) is dict:
            all_footnotes += footnotes["P"]["#text"]
        else:
            for i, item in enumerate(footnotes["P"]):
                if type(item) is str: 
                    if i == 0: all_footnotes += item
                    else: all_footnotes = all_footnotes + item
                else: 
                    if i == 0: all_footnotes += item["#text"]
                    else: all_footnotes = all_footnotes + item["#text"]
    return all_footnotes

def retrieve_useful_data_single(review):
    reduced_review = {}
    review = review["COCHRANE_REVIEW"]
    reduced_review["date"] = review["@MODIFIED"]
    
    try: quality_items = review["QUALITY_ITEMS"]["QUALITY_ITEM"]
    except: quality_items = None


    # INCLUDED
    if type(review["COVER_SHEET"]["TITLE"]) is str:
        reduced_review["title"] = review["COVER_SHEET"]["TITLE"]
    else: reduced_review["title"] = review["COVER_SHEET"]["TITLE"]["#text"]
    try: 
        reduced_review["included_studies"] = retrieve_studies_inner(
            review["STUDIES_AND_REFERENCES"]["STUDIES"]["INCLUDED_STUDIES"]["STUDY"],
            review["CHARACTERISTICS_OF_STUDIES"]["CHARACTERISTICS_OF_INCLUDED_STUDIES"]["INCLUDED_CHAR"],
            quality_items)
        reduced_review["included_footnotes"] = retrieve_footnotes(
            review["CHARACTERISTICS_OF_STUDIES"]["CHARACTERISTICS_OF_INCLUDED_STUDIES"]["FOOTNOTES"])
        reduced_review = retrieve_documents_study_type(reduced_review, "included_studies")
    except Exception as e: 
        reduced_review["included_studies"] = {}
    
    # EXCLUDED
    try: 
        reduced_review["excluded_studies"] = retrieve_studies_inner(
            review["STUDIES_AND_REFERENCES"]["STUDIES"]["EXCLUDED_STUDIES"]["STUDY"],
            review["CHARACTERISTICS_OF_STUDIES"]["CHARACTERISTICS_OF_EXCLUDED_STUDIES"]["EXCLUDED_CHAR"],
            None)
        reduced_review["included_footnotes"] = retrieve_footnotes(
                review["CHARACTERISTICS_OF_STUDIES"]["CHARACTERISTICS_OF_EXCLUDED_STUDIES"]["FOOTNOTES"])        
        reduced_review = retrieve_documents_study_type(reduced_review, "excluded_studies")
    except Exception as e: 
        reduced_review["excluded_studies"] = {}
    if len(reduced_review["included_studies"]) < minimum_amount_related_studies or len(reduced_review["excluded_studies"]) < minimum_amount_related_studies: return None
    return reduced_review

def preprocess_characteristics(char):
    char_method = ""
    char_participants = ""
    char_interventions = ""
    char_outcomes = ""
    char_notes = ""
    try:
        if type(char["CHAR_METHODS"]["P"]) is str: char_method += char["CHAR_METHODS"]["P"]
        elif type(char["CHAR_METHODS"]["P"]) is dict: char_method += char["CHAR_METHODS"]["P"]["#text"]
    except: pass
    
    try:
        if type(char["CHAR_PARTICIPANTS"]["P"]) is str: char_participants += char["CHAR_PARTICIPANTS"]["P"]
        elif type(char["CHAR_PARTICIPANTS"]["P"]) is dict: char_participants += char["CHAR_PARTICIPANTS"]["P"]["#text"]
    except: pass

    try:
        if type(char["CHAR_INTERVENTIONS"]["P"]) is str: char_interventions += char["CHAR_INTERVENTIONS"]["P"]
        elif type(char["CHAR_INTERVENTIONS"]["P"]) is dict: char_interventions += char["CHAR_INTERVENTIONS"]["P"]["#text"]
    except: pass

    try:
        if type(char["CHAR_OUTCOMES"]["P"]) is str: char_outcomes += char["CHAR_OUTCOMES"]["P"]
        elif type(char["CHAR_OUTCOMES"]["P"]) is dict: char_outcomes += char["CHAR_OUTCOMES"]["P"]["#text"]
    except: pass

    try:
        if type(char["CHAR_NOTES"]["P"]) is str: char_notes += char["CHAR_NOTES"]["P"]
        elif type(char["CHAR_NOTES"]["P"]) is dict: char_notes += char["CHAR_NOTES"]["P"]["#text"]
    except: pass

    try:
        if type(char["CHAR_REASON_FOR_EXCLUSION"]["P"]) is str: char_notes += char["CHAR_REASON_FOR_EXCLUSION"]["P"]
        elif type(char["CHAR_REASON_FOR_EXCLUSION"]["P"]) is dict: char_notes += char["CHAR_REASON_FOR_EXCLUSION"]["P"]["#text"]
    except: pass

    return {
        'method' : char_method,
        'participants' : char_participants,
        'interventions' : char_interventions,
        'outcomes' : char_outcomes,
        'notes' : char_notes
    }


def retrieve_studies_inner(studies, characteristics, quality_items):
    studies_reformat = {}
    if type(studies) != list:
        studies = [studies]
    for i, study in enumerate(studies):
        try:
            if type(study["REFERENCE"]) is list: reference = study["REFERENCE"][0]
            else: reference = study["REFERENCE"]        
            studies_reformat[str(i)] = get_subset_of_dict(reference)
            studies_reformat[str(i)]["id"] = study["@ID"]
            try: 
                char = characteristics[i]
                char = preprocess_characteristics(char)
            except: 
                char = None
            studies_reformat[str(i)]["characteristics"] = char
        except Exception as e: 
            print(e)

    if quality_items != None:
        for i, quality_item in enumerate(quality_items):
            quality_name = quality_item["NAME"]
            if quality_item["QUALITY_ITEM_DATA"] is None: continue
            if quality_item["QUALITY_ITEM_DATA"]["QUALITY_ITEM_DATA_ENTRY"] is None: continue
            if quality_item["QUALITY_ITEM_DATA"]["QUALITY_ITEM_DATA_ENTRY"] is dict: 
                quality_item["QUALITY_ITEM_DATA"]["QUALITY_ITEM_DATA_ENTRY"] = [quality_item["QUALITY_ITEM_DATA"]["QUALITY_ITEM_DATA_ENTRY"]]
            try:
                for item in quality_item["QUALITY_ITEM_DATA"]["QUALITY_ITEM_DATA_ENTRY"]:
                    if "@STUDY_ID" in item: study_id = item["@STUDY_ID"]
                    elif "_STUDY_ID" in item: study_id = item["_STUDY_ID"]
                    else: break
                    if "@RESULT" in item: result = item["@RESULT"]
                    elif "_RESULT" in item: result = item["_RESULT"]
                    else: break
                    for id, study in studies_reformat.items():
                        if study["id"] == study_id:
                            study[f"{quality_name}"] = result
                            break
            except Exception as e:
                print(e)
    return {k: v for k, v in studies_reformat.items() if v}

def calc_similarity_ratio(a, b):
    a = a.lower()
    a = re.sub('-', ' ', a)
    a = a.translate(str.maketrans('','',string.punctuation))
    a = re.sub(' +', ' ',a)
    b = b.lower()
    b = re.sub('-', ' ', b)
    b = b.translate(str.maketrans('','',string.punctuation))
    b = re.sub(' +', ' ', b)
    return SequenceMatcher(None, a, b).ratio()

def retrieve_abstract_and_date_from_xml(document):
    if document.abstract == None: return None
    return {
        "abstract" : document.abstract,
        "date" : convert_time_object(document.publication_date),
        "pubmed_id" : document.pubmed_id,
        "journal" : document.journal
    }

def retrive_year_date(document_date):
    year = None
    try:
        year = document_date.year
    except:
        year = None
    return str(year)

def find_correct_document(documents, included_study):
    best_document = None
    if 'AU' not in included_study or ('TI' not in included_study and 'SO' not in included_study): return None
    for document in documents:
        title_sim = calc_similarity_ratio(document.title, included_study['TI'])
        if title_sim < title_sim_ratio_2:
            continue
        if title_sim > title_sim_ratio:
            best_document = document
            break
        try: document_year = retrive_year_date(document.publication_date)
        except: document_year = None
        if document_year != None:
            year_diff = abs(int(document_year) - int(included_study['YR']))
            author_sim = calc_similarity_ratio(merge_authors(document.authors), included_study['AU'])
            if ((year_diff < 1) and (author_sim > author_sim_ratio) and (title_sim > title_sim_ratio_2)):
                best_document = document
                break
    return best_document

def parse_pubmed_doc(doc):
    xml_doc_to_string  = et.tostring(doc, encoding='UTF-8', method='xml')
    xml_parsed = xmltodict.parse(xml_doc_to_string)
    return xml_parsed

def search_pubmed_find_correct_document(query, included_study):
    try:
        documents = search_pubmed(query, pubmed_search_amount)
        if documents == None: return None
    except Exception as e:
        print(e)
    try:
        document = find_correct_document(documents, included_study)
        if document == None: return None
    except Exception as e:
        print(e)
    try:
        document_reduced = retrieve_abstract_and_date_from_xml(document)
    except Exception as e:
        print(e)
    return document_reduced
    

def create_document_query(included_study):
    if "TI" in included_study: return included_study["TI"]
    elif "SO" in included_study: return included_study["SO"]
    elif "AU" in included_study: return included_study["AU"]
    else: return None

def remove_articles_with_no_data(review, study_type):
    new_dict = review.copy()
    new_dict[study_type] = {}
    for study_id, included_study in review[study_type].items():
        if review[study_type][study_id]["document"] != None:
            new_dict[study_type][study_id] = included_study
    #print(f"Count {study_type} found in pubmed: {len(new_dict[study_type])}")
    return new_dict

def retrieve_documents_study_type(review, study_type):
    #print(f"Count {study_type}: {len(review[study_type])}")
    for study_id, included_study in review[study_type].items():
        query = create_document_query(included_study)
        if query == None: continue
        document = search_pubmed_find_correct_document(query, included_study)
        review[study_type][study_id]["document"] = document
    review = remove_articles_with_no_data(review, study_type)
    return review

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def study_type_to_rows(review, study_type):
    if study_type == "included_studies": label = 1
    else: label = 0
    all_rows = []
    for study_id, included_study in review[study_type].items():
        flattened = flatten(included_study)
        flattened["label"] = label
        flattened["date"] = review["date"]
        flattened["title"] = review["title"]
        all_rows.append(flattened)
    return all_rows

def to_pandas(reviews_filename):
    reviews = load_pickle(reviews_filename)
    all_rows = []
    all_df = None
    first = True
    print(f"Amount of reviews: {len(reviews)}")
    print("Dict to rows")
    for i, (review_name, review) in enumerate(reviews.items()):
        if i % 100 == 0: 
            print(f"Finished retrieving rows for review number {i}")
            print(f"Current total number of rows: {len(all_rows)}")
        if not review: continue
        all_rows += study_type_to_rows(review, "included_studies")
        all_rows += study_type_to_rows(review, "excluded_studies")
    print("Finished converting dict to rows")
    print(f"Totalt number of rows: {len(all_rows)}")
    print("Rows to pandas")
    '''
    for i, row in enumerate(all_rows):
        if i % 100 == 0: 
            print(f"Finished converting row number {i} to pandas")
        df = pd.DataFrame([row])
        if first: 
            all_df = df
            first = False
        else:
            all_df = pd.concat([all_df, df], axis=0, ignore_index=True)
    df = all_df
    '''
    df = pd.DataFrame.from_records(all_rows)
    try:
        df['date'] = pd.to_datetime(df['date'], utc=True)
    except:
        df['date'] = None
    df['study_date'] = pd.to_datetime(df['document_date'], utc=True)
    df.drop(['document_date'], axis=1)
    df['qid'] = df.groupby(['title']).ngroup()
    df.rename(columns={'document_abstract':'study_abstract'}, inplace=True)
    df.rename(columns={'document_pubmed_id':'pubmed_id'}, inplace=True)
    df.rename(columns={'document_journal':'journal'}, inplace=True)
    df['docid'] = df.groupby(['study_abstract']).ngroup()
    df.rename(columns={'TI':'study_title'}, inplace=True)
    df.rename(columns={'AU':'study_author'}, inplace=True)
    df.rename(columns={'YR':'study_year'}, inplace=True)
    df = df[df.columns[df.isnull().mean() < 0.7]]
    print("Finished converting dict to pandas")
    save_pickle(reviews_pandas_filename, df)

def merge_authors(authors):
    merged = ''
    for i, author in enumerate(authors):
        if author['lastname'] == None or author['firstname'] == None: continue
        if (i+1) != len(author): merged = merged + author['lastname'] + ' ' + author['firstname'] + ', '
        else: merged = merged + author['lastname'] + ' ' + author['firstname']
    return merged

def merge_author_and_title(authors, title):
    return authors + title

def reduce_and_save_df(df):
    df = df[df.columns[df.isnull().mean() < 0.7]]
    save_pickle(reviews_pandas_filename, df) 

def rename_df(df):
    df.drop(['document_date'], axis=1)
    df.rename(columns={'TI':'study_title'}, inplace=True)
    df.rename(columns={'AU':'study_author'}, inplace=True)
    df.rename(columns={'YR':'study_year'}, inplace=True)
    df.rename(columns={'id':'pubmed_id'}, inplace=True)
    save_pickle(reviews_pandas_filename, df) 

def rename_char_columns():
    df = load_pickle(reviews_pandas_filename)
    df.rename(columns={'characteristics_char_method':'study_method'}, inplace=True)
    df.rename(columns={'characteristics_char_participants':'study_participants'}, inplace=True)
    df.rename(columns={'characteristics_char_interventions':'study_interventions'}, inplace=True)
    df.rename(columns={'characteristics_char_outcomes':'study_outcomes'}, inplace=True)
    df.rename(columns={'characteristics_char_notes':'study_notes'}, inplace=True)
    save_pickle(reviews_pandas_filename, df) 

def save_load():
    df = load_pickle(reviews_pandas_filename)
    save_pickle(reviews_pandas_filename, df) 

def create_pairwise_single(q, df):
    new_data = []
    title = df['title'].iloc[0] 
    df_pos = df.loc[df['label'] == 1]
    df_neg = df.loc[df['label'] == 0]
    for index_1, row_pos in df_pos.iterrows():
        for index_2, row_neg in df_neg.iterrows():
            if random.random() < .5: #sample negative
                new_data.append(
                    [
                    q,
                    title,
                    row_pos["study_title"],
                    row_pos["study_abstract"],
                    row_neg["study_title"],
                    row_neg["study_abstract"],
                    row_pos["label"],
                    ]
                )
            else:
                new_data.append(
                    [
                    q,
                    title,
                    row_neg["study_title"],
                    row_neg["study_abstract"],
                    row_pos["study_title"],
                    row_pos["study_abstract"],
                    row_neg["label"],
                    ]
                )
    new_df = pd.DataFrame(new_data, columns = ["qid", "title", "pos_title", "pos_abstract", "neg_title", "neg_abstract", "label"])
    return new_df

def create_pairwise_method():
    new_dfs = []
    df = load_pickle(reviews_pandas_filename)
    qids =  df["qid"].unique().tolist()
    for q in qids:
        q_df = df.loc[df['qid'] == q]
        new_dfs.append(create_pairwise_single(q, q_df))
    new_dfs_complete = pd.concat(new_dfs)
    save_pickle("./data/processed/df_pairwise.pickle", new_dfs_complete) 

def create_approach_6_data():
    new_data = []
    df = load_pickle("./data/processed/df.pickle")
    print(f"Len of dataframe: {len(df)}")
    df = df[['qid', 'docid', 'title', 'study_title', 'study_abstract', 'label', 'date', 'study_date', 'pubmed_id']]
    df = df.drop_duplicates()
    print(f"Len of dataframe: {len(df)}")
    for i, (index_1, original_row) in enumerate(df.iterrows()):
        df_relevant = df.loc[df['qid'] == original_row['qid']]
        df_relevant = df_relevant.loc[df_relevant['label'] == 1]
        if len(df_relevant) == 0: continue
        relevant_abstract = df_relevant.loc[df_relevant.sample().index,'study_abstract'].to_numpy()[0]
        new_data.append(
            [
            original_row['qid'],
            original_row['docid'],
            original_row['title'],
            original_row['study_title'],
            original_row['study_abstract'],
            relevant_abstract,
            original_row['label'],
            original_row['date'],
            original_row['study_date'],
            original_row['pubmed_id'],
            ]
        ) 
        if i % 500 == 0: print(f"Finished number {i}")
    new_df = pd.DataFrame(new_data, columns = [
        "qid", "docid", "title", "study_title", "study_abstract", "relevant_abstract", "label", "date", "study_date", "pubmed_id"])
    df_splits = np.array_split(new_df, 3)
    for i, df_split in enumerate(df_splits):
        save_pickle(f"./data_processed/df_{i+1}.pickle", df_split) 

def write_top_20_to_csv():
    df = load_pickle("./data/processed/df_part_1.pickle")
    df = df[:20]
    df.to_csv('./data/analysis_testing/top_20.csv')

def merge_dfs():
    df_1 = load_pickle("./data/processed/df_part_1.pickle")
    df_2 = load_pickle("./data/processed/df_part_2.pickle")
    df = pd.concat([df_1,df_2])
    save_pickle("./data/processed/df.pickle", df)

def remove_rows_if_below_30():
    df_path = "./data_processed/"
    filenames = next(walk(df_path), (None, None, []))[2]
    dfs = []
    for file in filenames:
        dfs.append(load_pickle(df_path + file))
    df = pd.concat(dfs)
    counts = df['qid'].value_counts(dropna=False) 
    valids = counts[counts>=30].index

    df_reduced = df[df['qid'].isin(valids)]

    df_splits = np.array_split(df_reduced, 3)
    for i, df_split in enumerate(df_splits):
        save_pickle(f"./data_processed/df_{i+1}.pickle", df_split) 
    print("")

merge_dfs()
create_approach_6_data()
remove_rows_if_below_30()

#create_pairwise_method()

# This method preprocess all documents and store the processed objects in pickles
#preprocess_all_xml_documents(path = dir_path, n = None, reviews_filename = reviews_filename)

# This method convert the preprocessed documents to pandas objects
#to_pandas(reviews_filename = reviews_filename)

#rename_char_columns()
#save_load()
# This method load objects from pickles
#df = load_pickle(reviews_pandas_filename)
#value_counts = df.count()
#print("")
#rename_char_columns(df)

#write_top_20_to_csv()