from approach_1 import Approach1
from approach_2 import Approach2
from approach_3 import Approach3
from approach_4 import Approach4
from approach_5 import Approach5
from approach_6 import Approach6
from bm25 import Bm25
from pubmed import PubMedModel
from pytorch import init_device
from utils import get_avg_studies_pr_query, load_pickle, save_pickle, split_groupby
from sys import platform

import torch, gc
gc.collect()
torch.cuda.empty_cache()

class Main:
    def __init__(self, device, batch_size, n_epochs, reduced, verbose, 
    force_restart, bm25_search, pubmed_model_search_max, split_ratio, cleaning_types,
    early_stopping, freeze, dropout_value, df_path):
        if verbose:
            print("PRINTING CONFIG PARAMETERS")
            print(f"DEVICE: {device}")
            print(f"BATCH SIZE: {batch_size}")
            print(f"EPOCHS: {n_epochs}")
            print(f"REDUCED: {reduced}")
            print(f"FORCE RESTART: {force_restart}")
            print(f"BM25 INIT SEARCH: {bm25_search}")
            print(f"BM25 REDUCTION: {pubmed_model_search_max}")
            print(f"SPLIT RATIO: {split_ratio}")
            print(f"CLEANING TYPES: {cleaning_types}")
            print(f"FREEZE: {freeze}")
            print(f"DROPOUT VALUE: {dropout_value}")
        
        self.verbose = verbose

        '''
        if force_restart:
            df = load_pickle("./data/processed/df.pickle")
            self.train, self.val, self.test = split_groupby(df, split_ratio, 'qid', reduced, data_aug_n, self.verbose)
            save_pickle('./data/train_test/train_test.pickle', (self.train, self.val, self.test))
        else:
            self.train, self.val, self.test = load_pickle('./data/train_test/train_test.pickle')
        '''
        #df = load_pickle("./data/analysis_testing/df.pickle")
        df = load_pickle(df_path)
        #df = load_pickle("./data/processed/df_pairwise.pickle")
        self.train, self.val, self.test = split_groupby(df, split_ratio, 'qid', reduced, self.verbose)

        if self.verbose:
            print("Size of traing data: ", len(self.train))
            print("Size of validation data: ", len(self.val))
            print("Size of testing data: ", len(self.test))

        self.device = init_device(self.verbose, device)

        self.pubmed_model = PubMedModel(
            verbose = verbose,
            force_restart = True,
        )

        self.bm25 = Bm25(
            verbose = self.verbose, 
            pubmed_n = bm25_search, 
            reduced_n = pubmed_model_search_max,
        )
        
        self.approach1 = Approach1(
            bm25_search = bm25_search, 
            bm25_reduced = pubmed_model_search_max,
            force_restart = force_restart,
            verbose = verbose,
            n_epochs = n_epochs,
            cleaning_types = cleaning_types,
            batch_size = batch_size,
            device = self.device,
            early_stopping = early_stopping
        )

        self.approach2 = Approach2(
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose,
            device=self.device,
            force_restart=force_restart,
            early_stopping = early_stopping,
            freeze = freeze,
            dropout_value = dropout_value
        )

        self.approach3 = Approach3(
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose,
            device=self.device,
            force_restart=force_restart,
            early_stopping = early_stopping,
            freeze = freeze,
            dropout_value = dropout_value
        )

        self.approach4 = Approach4(
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose,
            device=self.device,
            force_restart=force_restart,
            early_stopping = early_stopping,
            freeze = freeze,
            dropout_value = dropout_value
        )

        self.approach5 = Approach5(
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose,
            device=self.device,
            force_restart=force_restart,
            early_stopping = early_stopping,
            freeze = freeze,
            dropout_value = dropout_value
        )

        self.approach6 = Approach6(
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose,
            device=self.device,
            force_restart=force_restart,
            early_stopping = early_stopping,
            freeze = freeze,
            dropout_value = dropout_value
        )

    def train_and_evaluate_all(self, text_combinations_approach_1_2, text_columns_approach_3_5, text_columns_approach_6, pre_trained_model_name, max_len, learning_rate, n_cv, max_abstract_length, bert_cv, pubmed_search_amount):

        if self.verbose:
            print(f"TEXT COMBINATIONS (APPROACH 1 and 2): {text_combinations_approach_1_2}")
            print(f"TEXT COLUMNS (APPROACH 3): {text_columns_approach_3_5}")
            print(f"TEXT COLUMNS (APPROACH 6): {text_columns_approach_6}")
            print(f"PRE TRAINED MODEL NAME: {pre_trained_model_name}")
            print(f"MAX LEN: {max_len}")
            print(f"LEARNING RATE: {learning_rate}")
            print(f"CV: {n_cv}")
        
        self.bm25.train_and_evaluate_model(
            'study_abstract',
            self.train,
            self.val,
            self.test,
            True)

        self.pubmed_model.evaluate(
            pubmed_search_amount = pubmed_search_amount, 
            df_test = self.test
        )
        '''
        self.approach1.train_and_evaluate_all(
            text_combinations = text_combinations_approach_1_2,
            train = self.train, 
            val = self.val,
            test = self.test,
            n_cv = n_cv,
            learning_rate = learning_rate
        )
        '''
        '''
        self.approach2.train_and_evaluate_all(
            text_combinations = text_combinations_approach_1_2,
            pre_trained_model_name = pre_trained_model_name,
            max_len = max_len,
            train = self.train,
            val = self.val,
            test = self.test,
            learning_rate = learning_rate,
            max_abstract_length = max_abstract_length,
            cv = bert_cv
        )
        '''
        '''
        self.approach3.train_and_evaluate_all(
            text_columns = text_columns_approach_3_5,
            pre_trained_model_name = pre_trained_model_name,
            max_len = max_len,
            train = self.train,
            val = self.val,
            test = self.test,
            learning_rate = learning_rate,
            max_abstract_length = max_abstract_length
        )
        '''
        '''
        self.approach6.train_all(
            text_columns = text_columns_approach_6,
            pre_trained_model_name = pre_trained_model_name,
            max_len = max_len,
            train = self.train,
            val = self.val,
            learning_rate = learning_rate,
            max_abstract_length = max_abstract_length,
            sub_approach = 1
        )
        '''
        '''
        self.approach6.train_all(
            text_columns = text_columns_approach_6,
            pre_trained_model_name = pre_trained_model_name,
            max_len = max_len,
            train = self.train,
            val = self.val,
            learning_rate = learning_rate,
            max_abstract_length = max_abstract_length,
            sub_approach = 2
        )
        '''
    
N_CV = 5
if platform == "linux":
    DEVICE = 'cuda'
    BATCH_SIZE = 8
else: 
    DEVICE = 'cpu'
    BATCH_SIZE = 2
DF_PATH = "./data_processed/df.pickle"
BM_25_INIT = 500
PUBMED_MODEL_SEARCH_MAX = 100
SPLIT_RATIO = 0.15
N_EPOCHS = 10
LEARNING_RATE = 0.0001
EARLY_STOPPING = 20
REDUCED = None
VERBOSE = True
FORCE_RESTART = False
MAX_LEN = 512
FREEZE = False
DROPOUT_VALUE = 0.3
MAX_ABSTRACT_LENGTH = None
CV_BERT = None
TEXT_COMBINATIONS_APPROACH_1_2 = ["title study_abstract", "title study_title study_abstract"]
TEXT_COLUMNS_APPROACH_3 = [
    "title",
    "study_title",
    "study_abstract"]
TEXT_COLUMNS_APPROACH_6 = {}
TEXT_COLUMNS_APPROACH_6[1] = [
    "title",
    "relevant_abstract",
    "study_abstract"]

TEXT_COLUMNS_APPROACH_6[2] = [
    "title study_abstract", "title relevant_abstract"]

PRE_TRAINED_MODEL_NAME =  {0: {'model_name': 'prajjwal1/bert-tiny', 'cleaning_type': False},
                           1: {'model_name': 'prajjwal1/bert-tiny', 'cleaning_type': True}}

CLEANING_TYPES = [True, False]

main = Main(
    device = DEVICE,
    batch_size=BATCH_SIZE,
    n_epochs = N_EPOCHS,
    force_restart = FORCE_RESTART,
    reduced = REDUCED,
    verbose = VERBOSE,
    bm25_search = BM_25_INIT,
    pubmed_model_search_max = PUBMED_MODEL_SEARCH_MAX,
    split_ratio = SPLIT_RATIO,
    cleaning_types = CLEANING_TYPES,
    early_stopping = EARLY_STOPPING,
    freeze = FREEZE,
    dropout_value = DROPOUT_VALUE,
    df_path = DF_PATH
)

main.train_and_evaluate_all(
    text_combinations_approach_1_2 = TEXT_COMBINATIONS_APPROACH_1_2,
    text_columns_approach_3_5 = TEXT_COLUMNS_APPROACH_3,
    text_columns_approach_6 = TEXT_COLUMNS_APPROACH_6,
    pre_trained_model_name = PRE_TRAINED_MODEL_NAME,
    max_len = MAX_LEN,
    learning_rate = LEARNING_RATE,
    n_cv = N_CV,
    max_abstract_length = MAX_ABSTRACT_LENGTH,
    bert_cv = CV_BERT,
    pubmed_search_amount = 1000
)

print("")