import torch
from data import create_data_loader
from models import Approach3Linear, Approach6_2Model, Approach6Model
from pubmed import search_and_process_pubmed
from pytorch import eval_model, init_device, predict_approach_3, train_approach6, train_general
from transformers import BertModel, BertTokenizer, AlbertModel, RobertaModel, AlbertTokenizer, RobertaTokenizer
from transformers import logging

from utils import create_dir_if_not_exists, find_best_model, get_model_type, get_tokenizer, preprocess_df_bert_3, preprocess_df_bert_6, save_pickle
logging.set_verbosity_error()

class BERT_Dual_Triple:
    def __init__(self, batch_size, n_epochs, verbose, device, force_restart, early_stopping, freeze, dropout_value):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.force_restart = force_restart
        self.device = device
        self.freeze = freeze
        self.dropout_value = dropout_value
        self.base_path1 = f"./models/BERT_Triple/"
        self.base_path2 = f"./models/BERT_Dual/"
        self.early_stopping = early_stopping
        create_dir_if_not_exists(self.base_path1)
        create_dir_if_not_exists(self.base_path2)
        
    
    def train_bert_model(self, text_columns, max_len, pre_trained_model_name, train, val, learning_rate, cleaning_type, max_abstract_length, sub_approach):
        if self.freeze == False: dropout_name = "None"
        else: dropout_name = self.dropout_value
        if sub_approach == 1:
            model_name = f'BERT-triple-representation-lr-{learning_rate}-clean-{cleaning_type}-batch-size-{self.batch_size}-max-abs-len-{max_abstract_length}-dropout-{dropout_name}'
        else:
            model_name = f'BERT-dual-interaction-lr-{learning_rate}-clean-{cleaning_type}-batch-size-{self.batch_size}-max-abs-len-{max_abstract_length}-dropout-{dropout_name}'
        if self.verbose: print(f"Training model: {model_name}")
        train = preprocess_df_bert_6(train, text_columns, cleaning_type, max_abstract_length, sub_approach)
        val = preprocess_df_bert_6(val, text_columns, cleaning_type, max_abstract_length, sub_approach)
        model_type = get_model_type(pre_trained_model_name)
        if sub_approach == 1: model = Approach6Model(1, pre_trained_model_name, self.freeze, self.dropout_value)
        else: model = Approach6_2Model(1, pre_trained_model_name, self.freeze, self.dropout_value)
        model.to(self.device)
        tokenizer = get_tokenizer(model_type, pre_trained_model_name)
        if sub_approach == 1: 
            temp_approach = 6
            class_name = "BERT_Triple"
        else: 
            temp_approach = 62
            class_name = "BERT_Dual"
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
                drop = self.dropout_value,
                class_name = class_name,
                max_abs_len = max_abstract_length,
                tf_idf=None)

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
        if sub_approach == 1:
            best_map, best_val_acc, best_model = find_best_model(["./history/BERT_Triple/"])
        else:
            best_map, best_val_acc, best_model = find_best_model(["./history/BERT_Dual/"])
        #self.evaluate(text_columns, best_model, max_len, test, learning_rate, cleaning_type, max_abstract_length, sub_approach)

    def find_best_model(self):
        best_map1, best_val_acc1, best_model1 = find_best_model(["./history/BERT_Triple/"])
        best_map2, best_val_acc2, best_model2 = find_best_model(["./history/BERT_Dual/"])
        if best_map1 > best_map2:
            return best_map1, best_val_acc1, best_model1
        else:
            return best_map2, best_val_acc2, best_model2

    def evaluate(self, model_name, max_len, test, sub_approach):
        if self.verbose: print(f"Evaluating model on test set: {model_name}")
        if sub_approach == 1:
            path = self.base_path1 + model_name +"/best_model.pt"
        else:
            path = self.base_path2 + model_name +"/best_model.pt"
        model, tokenizer, text_columns, cleaning_type, max_abstract_length = self.load_model(path, sub_approach)
        test = preprocess_df_bert_6(test, text_columns, cleaning_type, max_abstract_length, sub_approach)
        if sub_approach == 1:
            test_data_loader = create_data_loader(test, tokenizer, max_len, self.batch_size, 6, None, None, None, None)
            test_acc, test_loss, test_map_score = eval_model(6, model, test_data_loader, self.device, len(test_data_loader.dataset.labels))
        else:
            test_data_loader = create_data_loader(test, tokenizer, max_len, self.batch_size, 62, None, None, None, None)
            test_acc, test_loss, test_map_score = eval_model(62, model, test_data_loader, self.device, len(test_data_loader.dataset.labels))
        eval_result = {}
        eval_result['test_acc'] = test_acc
        eval_result['test_loss'] = test_loss
        eval_result['test_map_score'] = test_map_score
        eval_result['best_model'] = model_name
        print(f'Test   loss {test_loss}   accuracy {test_acc} map {test_map_score}')
        if sub_approach == 1:
            create_dir_if_not_exists("./eval/BERT_Triple/")
            save_pickle("./eval/BERT_Triple/eval", eval_result)
        else:
            create_dir_if_not_exists("./eval/BERT_Dual/")
            save_pickle("./eval/BERT_Dual/eval", eval_result)
            
    def load_model(self, path, sub_approach):
        checkpoint = torch.load(path)
        if sub_approach == 1:
            model = Approach6Model(1, checkpoint["pre_trained_model_name"], checkpoint["model_type"], checkpoint["drop"])
        else:
            model = Approach6_2Model(1, checkpoint["pre_trained_model_name"], checkpoint["model_type"], checkpoint["drop"])
        tokenizer = get_tokenizer(checkpoint["model_type"], checkpoint["pre_trained_model_name"])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        return model, tokenizer, checkpoint['text_combination'], checkpoint['cleaning_type'], checkpoint['max_abs_len']