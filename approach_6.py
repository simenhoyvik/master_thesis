import torch
from data import create_data_loader
from models import Approach3Linear, Approach6_2Model, Approach6Model
from pubmed import search_and_process_pubmed
from pytorch import eval_model, init_device, predict_approach_3, train_approach6, train_general
from transformers import BertModel, BertTokenizer, AlbertModel, RobertaModel, AlbertTokenizer, RobertaTokenizer
from transformers import logging

from utils import create_dir_if_not_exists, find_best_model, get_model_type, get_tokenizer, preprocess_df_bert_3, preprocess_df_bert_6, save_pickle
logging.set_verbosity_error()

class Approach6:
    def __init__(self, batch_size, n_epochs, verbose, device, force_restart, early_stopping, freeze, dropout_value):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.force_restart = force_restart
        self.device = device
        self.freeze = freeze
        self.dropout_value = dropout_value
        self.base_path = "/models/approach6/"
        self.path_model = self.base_path + "best_model"
        self.path_stats = self.base_path + "stats"
        self.early_stopping = early_stopping
        create_dir_if_not_exists(self.base_path)
    
    def train_bert_model(self, text_columns, max_len, pre_trained_model_name, train, val, learning_rate, cleaning_type, max_abstract_length, sub_approach):
        if "/" in pre_trained_model_name:
            split_pre_trained_model_name = pre_trained_model_name.split("/")
            pre_trained_model_name_path = split_pre_trained_model_name[-1]
        else:
            pre_trained_model_name_path = pre_trained_model_name
        model_name = f'{text_columns[sub_approach]}-{pre_trained_model_name_path}-{learning_rate}-cleaning-{cleaning_type}-batch-size-{self.batch_size}-max_abstract_len-{max_abstract_length}-dropout-{self.dropout_value}'
        if self.verbose: print(f"Training model: {model_name}")
        train = preprocess_df_bert_6(train, text_columns, cleaning_type, max_abstract_length, sub_approach)
        val = preprocess_df_bert_6(val, text_columns, cleaning_type, max_abstract_length, sub_approach)
        model_type = get_model_type(pre_trained_model_name)
        if sub_approach == 1: model = Approach6Model(1, pre_trained_model_name, self.freeze, self.dropout_value)
        else: model = Approach6_2Model(1, pre_trained_model_name, self.freeze, self.dropout_value)
        model.to(self.device)
        tokenizer = get_tokenizer(model_type, pre_trained_model_name)
        if sub_approach == 1: temp_approach = 6
        else: temp_approach = 62
        train_data_loader = create_data_loader(train, tokenizer, max_len, self.batch_size, temp_approach, None, None, None, None)
        val_data_loader = create_data_loader(val, tokenizer, max_len, self.batch_size, temp_approach, None, None, None, None)
        train_general(
                approach = temp_approach,
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
                text_combination = text_columns,
                cleaning_type = cleaning_type,
                batch_size = self.batch_size,
                pre_trained_model_name = pre_trained_model_name,
                model_type = model_type,
                drop = self.dropout_value)

    def train_all(self, text_columns, max_len, pre_trained_model_name, train, val, test, learning_rate, max_abstract_length, sub_approach):
        print("Training all models for Approach 6")
        for i, data in pre_trained_model_name.items():
            model_name = data['model_name']
            cleaning_type = data['cleaning_type']
            self.train_bert_model(
                text_columns = text_columns, 
                max_len = max_len,
                pre_trained_model_name = model_name, 
                train = train, 
                val = val, 
                learning_rate = learning_rate,
                cleaning_type = cleaning_type,
                max_abstract_length = max_abstract_length,
                sub_approach = sub_approach
            )
        best_map, best_model = find_best_model(["./history/approach6/"])
        self.evaluate(text_columns, best_model, max_len, test, learning_rate, cleaning_type, max_abstract_length, sub_approach)

    def evaluate(self, text_columns, model_name, max_len, test, learning_rate, cleaning_type, max_abstract_length, sub_approach):
        if self.verbose: print(f"Evaluating model on test set: {model_name}")
        path = "models/approach6/" + model_name +"/best_model.pt"
        model, tokenizer, text_columns, cleaning_type = self.load_model(path, sub_approach)
        test = preprocess_df_bert_6(test, text_columns, cleaning_type, max_abstract_length, sub_approach)
        test_data_loader = create_data_loader(test, tokenizer, max_len, self.batch_size, 6, None, None, None, None)
        test_acc, test_loss, test_map_score = eval_model(6, model, test_data_loader, self.device, len(test_data_loader.dataset.labels))
        eval_result = {}
        eval_result['test_acc'] = test_acc
        eval_result['test_loss'] = test_loss
        eval_result['test_map_score'] = test_map_score
        eval_result['best_model'] = model_name
        print(f'Test   loss {test_loss}   accuracy {test_acc} map {test_map_score}')
        save_pickle("./eval/approach6/eval", eval_result)
            
    def load_model(self, path, sub_approach):
        checkpoint = torch.load(path)
        model = Approach6Model(sub_approach, checkpoint["pre_trained_model_name"], checkpoint["model_type"], checkpoint["drop"])
        tokenizer = get_tokenizer(checkpoint["model_type"], checkpoint["pre_trained_model_name"])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        return model, tokenizer, checkpoint['text_combination'], checkpoint['cleaning_type']

    def predict(self, query, max_result):
        model,  tokenizer, text_combination, cleaning_type = self.load_model("models/approach2/title study_title study_abstract-albert-base-v2-0.0005/best_model.pt")
        test = search_and_process_pubmed(query, max_result)
        test = preprocess_df_bert_3(test, text_combination, cleaning_type)
        test_data_loader = create_data_loader(df=test, tokenizer=tokenizer, max_len=512, batch_size=self.batch_size, approach=2, text_columns=None, q_len=None)
        sorted = predict_approach_3(model, test_data_loader, test, self.device)
        return sorted