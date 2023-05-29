import os

import torch
from data import create_data_loader
from models import Approach1Model
from pytorch import eval_model, train_general
from utils import create_dir_if_not_exists, find_best_model, load_pickle, mean_average_precision_at_n, preprocess_df_1, save_pickle
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

    def train_sgd_pytorch(self, text_combination, cleaning, train, val, learning_rate):
        model_type = 'tf-idf-logreg'
        if 'study_title' in text_combination: model_name = f'QTA-{model_type}-{learning_rate}-cleaning-{cleaning}-batch-size-{self.batch_size}'
        else: model_name = f'QA-{model_type}-{learning_rate}-cleaning-{cleaning}-batch-size-{self.batch_size}'
        if self.verbose: print(f"Training model: {model_name}")
        train = preprocess_df_1(train, self.verbose,text_combination,cleaning)
        val = preprocess_df_1(val, self.verbose,text_combination,cleaning)
        tf_idf = TfidfVectorizer(min_df=8, ngram_range=(2,3))
        train_tf_idf = tf_idf.fit_transform(train['text'])
        val_tf_idf = tf_idf.transform(val['text'])
        train_data_loader = create_data_loader(train, None, None, self.batch_size, 1, None, None, None, train_tf_idf)
        val_data_loader = create_data_loader(val, None, None, self.batch_size, 1, None, None, None, val_tf_idf)
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
            model_type = model_type,
            class_name = self.__class__.__name__,
            max_abs_len = None,
            tf_idf = tf_idf)

    def train_and_evaluate_all(self, text_combinations, train, val, learning_rate):
        print("Training all models for Approach 1")
        for text_combination in text_combinations:
            for cleaning in self.cleaning_types:
                self.train_sgd_pytorch(text_combination, cleaning, train, val, learning_rate)
    
    def find_best_model(self):
        best_map, best_val_acc, best_model = find_best_model([f"./history/approach1/"])
        return best_map, best_val_acc, best_model
    
    def evaluate(self, model_name, max_len, test):
        if self.verbose: print(f"Evaluating model on test set: {model_name}")
        path = self.base_path + model_name +"/best_model.pt"
        model, text_combination, cleaning_type, tf_idf = self.load_model(path)
        test = preprocess_df_1(test, self.verbose, text_combination, cleaning_type)
        test_tf_idf = tf_idf.transform(test['text'])
        test_data_loader = create_data_loader(test, None, None, self.batch_size, 1, None, None, None, test_tf_idf)
        test_acc, test_loss, test_map_score = eval_model(1, model, test_data_loader, self.device, len(test_data_loader.dataset.labels))
        eval_result = {}
        eval_result['test_acc'] = test_acc
        eval_result['test_loss'] = test_loss
        eval_result['test_map_score'] = test_map_score
        eval_result['best_model'] = model_name
        print(f'Test   loss {test_loss}   accuracy {test_acc} map {test_map_score}')
        create_dir_if_not_exists(f"./eval/{self.__class__.__name__}/")
        save_pickle(f"./eval/{self.__class__.__name__}/eval", eval_result)

    def load_model(self, path):
        checkpoint = torch.load(path)
        model = Approach1Model(n_classes = 1, tf_idf_size = checkpoint['tf_idf'].shape[1], dropout_value = None)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        return model, checkpoint['text_combination'], checkpoint['cleaning_type'], checkpoint['tf_idf']