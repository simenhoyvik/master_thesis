import os
from data import create_data_loader
from models import Approach1Model
from pytorch import train_general
from utils import create_dir_if_not_exists, load_pickle, mean_average_precision_at_n, preprocess_df_1
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
pd.set_option("max_colwidth", None)

class TFIDF_LOGREG:
    def __init__(self, bm25_search, bm25_reduced, force_restart, verbose, cleaning_types, n_epochs, batch_size, device, early_stopping):
        self.bm25_search = bm25_search
        self.bm25_reduced = bm25_reduced
        self.force_restart = force_restart
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.cleaning_types = cleaning_types
        self.batch_size = batch_size
        self.device = device
        self.early_stopping = early_stopping
        self.models = {}
        self.base_path = "./models/approach1/"
        create_dir_if_not_exists(self.base_path)

    def train_sgd_pytorch(self, text_combination, cleaning, train, val, test, learning_rate):
        model_type = 'tf-idf-logreg'
        model_name = f'{text_combination}-{model_type}-{learning_rate}-cleaning-{cleaning}-batch-size-{self.batch_size}'
        if self.verbose: print(f"Training model: {model_name}")
        train = preprocess_df_1(train, self.verbose,text_combination,cleaning)
        val = preprocess_df_1(val, self.verbose,text_combination,cleaning)
        test = preprocess_df_1(test, self.verbose,text_combination,cleaning)
        tf_idf = TfidfVectorizer(min_df=8, ngram_range=(2,3))
        train_tf_idf = tf_idf.fit_transform(train['text'])
        val_tf_idf = tf_idf.transform(val['text'])
        test_tf_idf = tf_idf.transform(test['text'])
        train_data_loader = create_data_loader(train, None, None, self.batch_size, 1, None, None, None, train_tf_idf)
        val_data_loader = create_data_loader(val, None, None, self.batch_size, 1, None, None, None, val_tf_idf)
        test_data_loader = create_data_loader(test, None, None, self.batch_size, 1, None, None, None, test_tf_idf)
        model = Approach1Model(n_classes = 1, tf_idf_size = train_tf_idf.shape[1], dropout_value = None)
        model.to(self.device)
        train_general(
            approach = 1,
            model = model,
            tokenizer = None,
            train_data_loader = train_data_loader,
            val_data_loader = val_data_loader,
            device = self.device, 
            N_EPOCHS = self.n_epochs,
            verbose = self.verbose,
            learning_rate = learning_rate,
            model_name = model_name,
            early_stopping = self.early_stopping,
            text_combination = text_combination,
            cleaning_type = cleaning,
            batch_size = self.batch_size,
            pre_trained_model_name = 'None',
            model_type = model_type)

    def train_and_evaluate_all(self, text_combinations, train, val, test, n_cv, learning_rate):
        print("Training all models for Approach 1")
        for text_combination in text_combinations:
            for cleaning in self.cleaning_types:
                #self.train_sgd_sklearn(text_combination, cleaning, train, val, n_cv)
                self.train_sgd_pytorch(text_combination, cleaning, train, val, test, learning_rate)

    def evaluate_best(self):
        best_model = load_pickle(self.path_model)
        best_params, best_score, best_type, best_cleaning, best_split = load_pickle(self.path_stats)
        _, test = load_pickle(f"./data/train_test/approach1/data_{best_split}")

        #test = prepare_df_based_on_parameters(test, best_type, best_cleaning)

        print("\nBEST MODEL: ")
        print(f"Model - Type: {best_type}, Cleaning: {best_cleaning}, Split: {best_split}")
        print(f"Best Result: {best_score}")
        print(f"Best Parameters: {best_params}\n")

        mean_average_precision_at_n(best_model, test, 5)
        mean_average_precision_at_n(best_model, test, 7)
        mean_average_precision_at_n(best_model, test, 10)
        mean_average_precision_at_n(best_model, test, 15)

        #evaluate_acc_model(model, X_test, Y_test, "sgd_model")
        #evaluate_clf_report(model, X_test, Y_test)
        #explain_evaluation(model, X_test, Y_test)
        #print_top_bottom_idf(model)
        #print_tfidf_document(self.sgd_model, self.X_test)
        #evaluate_acc_top_n(self.sgd_model, self.X_test, self.Y_test, 20)

    #def predict(self, query = None):
    #    best_model = load_pickle(self.path_model)
    #    if query == None: query = sample_random_query(self.df, "title", self.verbose)
    #    result_df = search_pubmed_reduce(query, self.bm25_search, self.bm25_reduced, self.verbose)
    #    sorted_df = sort_documents(best_model, result_df)
    #    return sorted_df