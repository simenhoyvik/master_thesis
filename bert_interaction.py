import os
import pandas as pd
import torch
from data import create_data_loader
from models import BinaryClassifier
from pytorch import eval_model, train_general
from utils import create_dir_if_not_exists, find_best_model, get_model_type, get_tokenizer, preprocess_df_bert_2_4, save_pickle

import torch, gc
gc.collect()
torch.cuda.empty_cache()

class BERT_Interaction:
    def __init__(self, batch_size, n_epochs, verbose, device, force_restart, early_stopping, freeze, dropout_value):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.force_restart = force_restart
        self.device = device
        self.freeze = freeze
        self.dropout_value = dropout_value
        self.base_path = f"./models/{self.__class__.__name__}/"
        self.early_stopping = early_stopping
        create_dir_if_not_exists(self.base_path)

    def train_bert_model(self, text_combination, max_len, pre_trained_model_name, train, val, learning_rate, cleaning_type, max_abstract_length):
        if self.freeze == False: dropout_name = "None"
        else: dropout_name = self.dropout_value
        if 'study_title' in text_combination: model_name = f'BERT-interaction-QTA-lr-{learning_rate}-clean-{cleaning_type}-batch-size-{self.batch_size}-max-abs-len-{max_abstract_length}-dropout-{dropout_name}'
        else: model_name = f'BERT-interaction-QA-lr-{learning_rate}-clean-{cleaning_type}-batch-size-{self.batch_size}-max-abs-len-{max_abstract_length}-dropout-{dropout_name}'
        if self.verbose: print(f"Training model: {model_name}")
        model_type = get_model_type(pre_trained_model_name)
        model = BinaryClassifier(1, pre_trained_model_name, model_type, self.freeze, self.dropout_value)
        model.to(self.device)
        tokenizer = get_tokenizer(model_type, pre_trained_model_name)
        train = preprocess_df_bert_2_4(train, text_combination, cleaning_type, max_abstract_length, 'text')
        val = preprocess_df_bert_2_4(val, text_combination, cleaning_type, max_abstract_length, 'text')
        train_data_loader = create_data_loader(train, tokenizer, max_len, self.batch_size, 2, None, None, None, None)
        val_data_loader = create_data_loader(val, tokenizer, max_len, self.batch_size, 2, None, None, None, None)
        train_general(
            approach = 2,
            model = model,
            tokenizer = tokenizer,
            train_data_loader = train_data_loader,
            val_data_loader = val_data_loader,
            device = self.device, 
            N_EPOCHS = self.n_epochs,
            verbose = self.verbose,
            learning_rate = learning_rate,
            model_name = model_name,
            early_stopping = self.early_stopping,
            text_combination = text_combination,
            cleaning_type = cleaning_type,
            batch_size = self.batch_size,
            pre_trained_model_name = pre_trained_model_name,
            model_type = model_type,
            drop = self.dropout_value,
            class_name = self.__class__.__name__,
            max_abs_len = max_abstract_length,
            tf_idf=None)

    def train_and_evaluate_all(self, text_combinations, pre_trained_model_name, max_len, train, val, learning_rate, max_abstract_length):
        print("Training all models for Approach 2")
        for text_combination in text_combinations:
            for i, data in pre_trained_model_name.items():
                model_name = data['model_name']
                cleaning_type = data['cleaning_type']
                self.train_bert_model(
                    text_combination = text_combination,
                    max_len = max_len,
                    pre_trained_model_name = model_name,
                    train = train,
                    val = val,
                    learning_rate = learning_rate,
                    cleaning_type=cleaning_type,
                    max_abstract_length = max_abstract_length
                )
        #best_map, best_val_acc, best_model = find_best_model([f"./history/{self.__class__.__name__}/"])
        #self.evaluate(text_combination, best_model, max_len, test, learning_rate, cleaning_type, max_abstract_length)

    def find_best_model(self):
        best_map, best_val_acc, best_model = find_best_model([f"./history/{self.__class__.__name__}/"])
        return best_map, best_val_acc, best_model
    
    def evaluate(self, model_name, max_len, test):
        if self.verbose: print(f"Evaluating model on test set: {model_name}")
        path = self.base_path + model_name +"/best_model.pt"
        model, tokenizer, text_combination, cleaning_type, max_abstract_length = self.load_model(path)
        test = preprocess_df_bert_2_4(test, text_combination, cleaning_type, max_abstract_length, 'text')
        test_data_loader = create_data_loader(test, tokenizer, max_len, self.batch_size, 2, None, None, None, None)
        test_acc, test_loss, test_map_score = eval_model(2, model, test_data_loader, self.device, len(test_data_loader.dataset.labels))
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
        model = BinaryClassifier(1, checkpoint["pre_trained_model_name"], checkpoint["model_type"], True, self.dropout_value)
        tokenizer = get_tokenizer(checkpoint["model_type"], checkpoint["pre_trained_model_name"])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        checkpoint['max_abs_len'] = None
        return model, tokenizer, checkpoint['text_combination'], checkpoint['cleaning_type'], checkpoint['max_abs_len']