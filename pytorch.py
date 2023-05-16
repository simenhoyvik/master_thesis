import numpy as np
from torch import nn
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam
import warnings
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup, AdamW
import torch
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader
from data import create_data_loader

from utils import all_items_same, average_scores, calc_map, create_dir_if_not_exists, find_split_in_list, init_qid_dict, save_pickle, to_url
warnings.simplefilter("ignore", UserWarning)

def init_device(verbose, device):
    if device == 'cpu': return torch.device('cpu')
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Training on device {device}")
    return device

def save_checkpoint(epoch, model, tokenizer, map, filename, pre_trained_model_name, model_type, text_combination, cleaning_type, drop):
    print('Save PyTorch model to {}'.format(filename))
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'map': map,
        'pre_trained_model_name' : pre_trained_model_name,
        'model_type' : model_type,
        'text_combination' : text_combination,
        'cleaning_type' : cleaning_type,
        'drop' : drop
    }
    torch.save(state, filename)

def load_checkpoint(filename, device='cpu'):
    print('Load PyTorch model from {}'.format(filename))
    state = torch.load(filename, map_location='cpu') if device == 'cpu' else torch.load(filename)
    return state['epoch'], state['model'], state['tokenizer'], state['map']

def init_optimizer(model, learning_rate, train_data_loader, N_EPOCHS):
    optimizer = AdamW(model.parameters(), lr=learning_rate, no_deprecation_warning=True, weight_decay=0.01, betas=(0.9, 0.999))
    #total_steps = len(train_data_loader) * N_EPOCHS
    num_training_steps = len(train_data_loader) * N_EPOCHS
    num_warmup_steps = int(num_training_steps * 0.2)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    return optimizer, scheduler

def train_approach2_cv(approach, model, tokenizer, train, device, N_EPOCHS, verbose, learning_rate, batch_size, model_name, early_stopping, pre_trained_model_name, model_type, text_combination, cleaning_type, cv, max_len):
    torch.cuda.empty_cache()
    if verbose: print("Starting training") 

    model_path_dir = f'models/approach{approach}/{model_name}'
    base_history_path = f'history/approach{approach}/{model_name}'
    create_dir_if_not_exists(model_path_dir)
    create_dir_if_not_exists(base_history_path)
    best_model_path = model_path_dir + "/best_model.pt"
    history_path = base_history_path + "/history.pickle"

    history = {}
    best_map = None
    best_loss = None
    loss_not_decreasing_counter = 0
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    splits = KFold(n_splits = cv, shuffle = True, random_state = 42)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train)))):

        print('Fold {}'.format(fold + 1))
        history[fold] = defaultdict(list)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_data_loader = create_data_loader(train, tokenizer, max_len, batch_size, 2, None, None, train_sampler, None)
        val_data_loader = create_data_loader(train, tokenizer, max_len, batch_size, 2, None, None, val_sampler, None)

        optimizer, scheduler = init_optimizer(model, learning_rate, train_data_loader, N_EPOCHS)

        for epoch in range(N_EPOCHS):

            if verbose: 
                print(f'Epoch {epoch + 1}/{N_EPOCHS}')
                print('-' * 20)

            train_acc, train_loss, train_map_score = train_epoch(
                approach,
                model,
                train_data_loader,    
                loss_fn, 
                optimizer, 
                device,
                scheduler,
                len(train_sampler.indices)
            )

            val_acc, val_loss, val_map_score = eval_model(
                approach,
                model,
                val_data_loader,
                device, 
                len(val_sampler.indices)
            )

            if verbose: 
                print(f'Train loss {train_loss} accuracy {train_acc} map {train_map_score}')
                print(f'Val   loss {val_loss}   accuracy {val_acc}   map {val_map_score}')

            history[fold]['train_acc'].append(train_acc)
            history[fold]['train_loss'].append(train_loss)
            history[fold]['train_map'].append(train_map_score)
            history[fold]['val_acc'].append(val_acc)
            history[fold]['val_loss'].append(val_loss)
            history[fold]['val_map'].append(val_map_score)

            if best_map is None or val_map_score > best_map:
                save_checkpoint(epoch, model, tokenizer, best_map, best_model_path, pre_trained_model_name, model_type, text_combination, cleaning_type, drop)
                best_map = val_map_score
    
    history = average_scores(history, N_EPOCHS, cv)
    
    save_pickle(history_path, history)
    if verbose: print("Finished training")

def predict_approach_2_4(model, data_loader, test, device):
    df = pd.DataFrame(columns=['pred', 'pubmed_id'])
  
    for i, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        qids = d["qids"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        outputs_numpy = outputs.cpu().detach().numpy()

        combined = np.c_[ outputs_numpy, qids ] 
        combined = pd.DataFrame(combined, columns=["pred", "pubmed_id"])
        df = pd.concat([df, combined], ignore_index=True)

    sorted = df.sort_values(['pred'], ascending=False)
    sorted = pd.merge(sorted, test, left_on='pubmed_id', right_on='pubmed_id', how='left')
    sorted['url'] = sorted['pubmed_id'].apply(to_url)
    sorted = sorted[['index', 'study_title', 'study_abstract', 'url']]
    to_dict = sorted.to_dict('records')
    return to_dict

def predict_approach_3(model, data_loader, test, device):
    df = pd.DataFrame(columns=['pred', 'pubmed_id'])
  
    for i, d in enumerate(data_loader):
        title_encoding = d['encoding_title']
        study_title_encoding = d['encoding_study_title']
        study_abstract_encoding = d['encoding_study_abstract']

        input_ids_title = title_encoding["input_ids"].to(device)
        input_ids_study_title = study_title_encoding["input_ids"].to(device)
        input_ids_study_abstract = study_abstract_encoding["input_ids"].to(device)

        attention_mask_title = title_encoding["attention_mask"].to(device)
        attention_mask_study_title = study_title_encoding["attention_mask"].to(device)
        attention_mask_study_abstract = study_abstract_encoding["attention_mask"].to(device)

        token_type_ids_title = title_encoding["token_type_ids"].to(device)
        token_type_ids_study_title = study_title_encoding["token_type_ids"].to(device)
        token_type_ids_study_abstract = study_abstract_encoding["token_type_ids"].to(device)

        qids = d["qids"].numpy()

        outputs = model(input_ids_title, attention_mask_title, token_type_ids_title,
                        input_ids_study_title, attention_mask_study_title, token_type_ids_study_title,
                        input_ids_study_abstract, attention_mask_study_abstract, token_type_ids_study_abstract)

        outputs_numpy = outputs.cpu().detach().numpy()

        combined = np.c_[ outputs_numpy, qids ] 
        combined = pd.DataFrame(combined, columns=["pred", "pubmed_id"])
        df = pd.concat([df, combined], ignore_index=True)

    sorted = df.sort_values(['pred'], ascending=False)
    sorted = pd.merge(sorted, test, left_on='pubmed_id', right_on='pubmed_id', how='left')
    sorted['url'] = sorted['pubmed_id'].apply(to_url)
    sorted = sorted[['index', 'study_title', 'study_abstract', 'url']]
    to_dict = sorted.to_dict('records')
    return to_dict

def load_data_and_forward_model(approach, d, device, model):
    if approach == 1:
        inputs = d["tf_idf_matrix"].to(device)
        inputs = inputs.to(torch.float32)
        outputs = model(inputs=inputs)
    elif approach == 2:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    elif approach == 3:
        title_encoding = d['encoding_title']
        study_title_encoding = d['encoding_study_title']
        study_abstract_encoding = d['encoding_study_abstract']

        input_ids_title = title_encoding["input_ids"].to(device)
        input_ids_study_title = study_title_encoding["input_ids"].to(device)
        input_ids_study_abstract = study_abstract_encoding["input_ids"].to(device)

        attention_mask_title = title_encoding["attention_mask"].to(device)
        attention_mask_study_title = study_title_encoding["attention_mask"].to(device)
        attention_mask_study_abstract = study_abstract_encoding["attention_mask"].to(device)

        token_type_ids_title = title_encoding["token_type_ids"].to(device)
        token_type_ids_study_title = study_title_encoding["token_type_ids"].to(device)
        token_type_ids_study_abstract = study_abstract_encoding["token_type_ids"].to(device)

        outputs = model(input_ids_title, attention_mask_title, token_type_ids_title,
                        input_ids_study_title, attention_mask_study_title, token_type_ids_study_title,
                        input_ids_study_abstract, attention_mask_study_abstract, token_type_ids_study_abstract)
    elif approach == 4:
        query_encoding = d['query']
        document_encoding = d['document']
        
        query_tokens = query_encoding['tokens'].to(device)
        document_tokens = document_encoding['tokens'].to(device)

        query_mask = query_encoding['mask'].to(device)
        document_mask = document_encoding['mask'].to(device)

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            query_tok = query_tokens,
            doc_tok = document_tokens,
            query_mask = query_mask,
            document_mask = document_mask
        )
    elif approach == 5:
        q_embeddings = d['encoding_query']['embeddings'].to(device)
        q_embeddings_ids = d['encoding_query']['embeddings_ids'].to(device)
        d_embeddings = d['encodings_document']['embeddings'].to(device)
        d_embeddings_ids = d['encodings_document']['embeddings_ids'].to(device)

        outputs = model(q_embeddings, d_embeddings, q_embeddings_ids, d_embeddings_ids)
    elif approach == 6:
        title_encoding = d['encoding_title']
        study_abstract_encoding = d['encoding_study_abstract']
        relevant_document_encoding = d['encoding_relevant_document']

        input_ids_title = title_encoding["input_ids"].to(device)
        input_ids_study_abstract = study_abstract_encoding["input_ids"].to(device)
        input_ids_relevant_document = relevant_document_encoding["input_ids"].to(device)

        attention_mask_title = title_encoding["attention_mask"].to(device)
        attention_mask_study_abstract = study_abstract_encoding["attention_mask"].to(device)
        attention_mask_relevant_document = relevant_document_encoding["attention_mask"].to(device)

        token_type_ids_title = title_encoding["token_type_ids"].to(device)
        token_type_ids_study_abstract = study_abstract_encoding["token_type_ids"].to(device)
        token_type_ids_relevant_document = relevant_document_encoding["token_type_ids"].to(device)

        outputs = model(input_ids_title, attention_mask_title, token_type_ids_title,
                        input_ids_study_abstract, attention_mask_study_abstract, token_type_ids_study_abstract,
                        input_ids_relevant_document, attention_mask_relevant_document, token_type_ids_relevant_document)
        
    elif approach == 62:
        texts_1_encoding = d['encoding_text_1']
        texts_2_encoding = d['encoding_text_2']

        input_ids_1 = texts_1_encoding["input_ids"].to(device)
        input_ids_2 = texts_2_encoding["input_ids"].to(device)

        attention_mask_1 = texts_1_encoding["attention_mask"].to(device)
        attention_mask_2 = texts_2_encoding["attention_mask"].to(device)

        token_type_ids_1 = texts_1_encoding["token_type_ids"].to(device)
        token_type_ids_2 = texts_2_encoding["token_type_ids"].to(device)

        outputs = model(input_ids_1, attention_mask_1, token_type_ids_1,
                        input_ids_2, attention_mask_2, token_type_ids_2)
        
    targets = d["targets"].to(device)
    qids = d["qids"].numpy()

    return outputs, targets, qids

def train_epoch(approach, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])
  
    for i, d in enumerate(data_loader):
        outputs, targets, qids = load_data_and_forward_model(approach, d, device, model)

        probs = torch.sigmoid(outputs)
        preds = (probs>0.5).type(torch.int)
        loss = loss_fn(outputs.squeeze(-1), targets.float())

        targets_numpy = targets.cpu().detach().numpy()
        outputs_numpy = outputs.cpu().detach().numpy()
        
        combined = np.c_[ outputs_numpy, qids ]
        combined = np.c_[ combined, targets_numpy ]
        combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

        df = pd.concat([df, combined], ignore_index=True)

        correct_predictions += torch.sum(preds.squeeze(-1) == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i % 10 == 0: print(f"Finished Batch {i+1} - Loss {losses[-1]}")

    map_score = calc_map(df)
    
    return correct_predictions.double() / n_examples, np.mean(losses), map_score

def train_epoch_approach_6(data_loader, loss_fn, optimizer, device, scheduler, n_examples, model):
    model = model.train()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])
  
    for i, d in enumerate(data_loader):
        title_encoding = d['encoding_title']
        study_title_encoding = d['encoding_study_title']
        study_abstract_encoding = d['encoding_study_abstract']
        relevant_document_encoding = d['encoding_relevant_document']

        input_ids_title = title_encoding["input_ids"].to(device)
        input_ids_study_title = study_title_encoding["input_ids"].to(device)
        input_ids_study_abstract = study_abstract_encoding["input_ids"].to(device)
        input_ids_relevant_document = relevant_document_encoding["input_ids"].to(device)

        attention_mask_title = title_encoding["attention_mask"].to(device)
        attention_mask_study_title = study_title_encoding["attention_mask"].to(device)
        attention_mask_study_abstract = study_abstract_encoding["attention_mask"].to(device)
        attention_mask_relevant_document = relevant_document_encoding["attention_mask"].to(device)

        token_type_ids_title = title_encoding["token_type_ids"].to(device)
        token_type_ids_study_title = study_title_encoding["token_type_ids"].to(device)
        token_type_ids_study_abstract = study_abstract_encoding["token_type_ids"].to(device)
        token_type_ids_relevant_document = relevant_document_encoding["token_type_ids"].to(device)

        targets = d["targets"].to(device)
        qids = d["qids"].numpy()

        outputs = model(input_ids_title, attention_mask_title, token_type_ids_title,
                        input_ids_study_title, attention_mask_study_title, token_type_ids_study_title,
                        input_ids_study_abstract, attention_mask_study_abstract, token_type_ids_study_abstract,
                        input_ids_relevant_document, attention_mask_relevant_document, token_type_ids_relevant_document)

        probs = torch.sigmoid(outputs)
        preds = (probs>0.5).type(torch.int)
        loss = loss_fn(outputs.squeeze(-1), targets.float())

        targets_numpy = targets.cpu().detach().numpy()
        outputs_numpy = outputs.cpu().detach().numpy()

        combined = np.c_[ outputs_numpy, qids ]
        combined = np.c_[ combined, targets_numpy ]
        combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

        df = pd.concat([df, combined], ignore_index=True)

        correct_predictions += torch.sum(preds.squeeze(-1) == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i % 10 == 0: print(f"Finished Batch {i+1} - Loss {losses[-1]}")

    map_score = calc_map(df)
    
    return correct_predictions.double() / n_examples, np.mean(losses), map_score

def eval_model(approach, model, data_loader, device, n_examples):
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    model = model.eval()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])

    with torch.no_grad():
        for d in data_loader:
            outputs, targets, qids = load_data_and_forward_model(approach, d, device, model)
            
            probs = torch.sigmoid(outputs)
            preds = (probs>0.5).type(torch.int)
            loss = loss_fn(outputs.squeeze(-1), targets.float())

            targets_numpy = targets.cpu().detach().numpy()
            outputs_numpy = outputs.cpu().detach().numpy()

            combined = np.c_[ outputs_numpy, qids ] 
            combined = np.c_[ combined, targets_numpy ] 
            combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

            df = pd.concat([df, combined], ignore_index=True)

            correct_predictions += torch.sum(preds.squeeze(-1) == targets)
            losses.append(loss.item())

    map_score = calc_map(df)

    return correct_predictions.double() / n_examples, np.mean(losses), map_score


def eval_model_approach_4(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])

    with torch.no_grad():
        for d in data_loader:
            query_encoding = d['query']
            document_encoding = d['document']
            
            query_tokens = query_encoding['tokens'].to(device)
            document_tokens = document_encoding['tokens'].to(device)

            query_mask = query_encoding['mask'].to(device)
            document_mask = document_encoding['mask'].to(device)

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets = d["targets"].to(device)
            qids = d["qids"].numpy()

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                query_tok = query_tokens,
                doc_tok = document_tokens,
                query_mask = query_mask,
                document_mask = document_mask
            )

            probs = torch.sigmoid(outputs)
            preds = (probs>0.5).type(torch.int)
            loss = loss_fn(outputs.squeeze(-1), targets.float())

            targets_numpy = targets.cpu().detach().numpy()
            outputs_numpy = outputs.cpu().detach().numpy()

            combined = np.c_[ outputs_numpy, qids ] 
            combined = np.c_[ combined, targets_numpy ] 
            combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

            df = pd.concat([df, combined], ignore_index=True)

            correct_predictions += torch.sum(preds.squeeze(-1) == targets)
            losses.append(loss.item())

    map_score = calc_map(df)

    return correct_predictions.double() / n_examples, np.mean(losses), map_score


def eval_model_approach_6(data_loader, loss_fn, device, n_examples, model):
    model = model.eval()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])

    with torch.no_grad():
        for d in data_loader:
            title_encoding = d['encoding_title']
            study_title_encoding = d['encoding_study_title']
            study_abstract_encoding = d['encoding_study_abstract']
            relevant_document_encoding = d['encoding_relevant_document']

            input_ids_title = title_encoding["input_ids"].to(device)
            input_ids_study_title = study_title_encoding["input_ids"].to(device)
            input_ids_study_abstract = study_abstract_encoding["input_ids"].to(device)
            input_ids_relevant_document = relevant_document_encoding["input_ids"].to(device)

            attention_mask_title = title_encoding["attention_mask"].to(device)
            attention_mask_study_title = study_title_encoding["attention_mask"].to(device)
            attention_mask_study_abstract = study_abstract_encoding["attention_mask"].to(device)
            attention_mask_relevant_document = relevant_document_encoding["attention_mask"].to(device)

            token_type_ids_title = title_encoding["token_type_ids"].to(device)
            token_type_ids_study_title = study_title_encoding["token_type_ids"].to(device)
            token_type_ids_study_abstract = study_abstract_encoding["token_type_ids"].to(device)
            token_type_ids_relevant_document = relevant_document_encoding["token_type_ids"].to(device)

            targets = d["targets"].to(device)
            qids = d["qids"].numpy()

            outputs = model(input_ids_title, attention_mask_title, token_type_ids_title,
                            input_ids_study_title, attention_mask_study_title, token_type_ids_study_title,
                            input_ids_study_abstract, attention_mask_study_abstract, token_type_ids_study_abstract,
                            input_ids_relevant_document, attention_mask_relevant_document, token_type_ids_relevant_document)
            
            probs = torch.sigmoid(outputs)
            preds = (probs>0.5).type(torch.int)
            loss = loss_fn(outputs.squeeze(-1), targets.float())

            targets_numpy = targets.cpu().detach().numpy()
            outputs_numpy = outputs.cpu().detach().numpy()

            combined = np.c_[ outputs_numpy, qids ] 
            combined = np.c_[ combined, targets_numpy ] 
            combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

            df = pd.concat([df, combined], ignore_index=True)

            correct_predictions += torch.sum(preds.squeeze(-1) == targets)
            losses.append(loss.item())

    map_score = calc_map(df)

    return correct_predictions.double() / n_examples, np.mean(losses), map_score

def eval_drmm(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])

    with torch.no_grad():
        for d in data_loader:
            q_embeddings = d['encoding_query']['embeddings'].to(device)
            q_embeddings_ids = d['encoding_query']['embeddings_ids'].to(device)
            d_embeddings = d['encodings_document']['embeddings'].to(device)
            d_embeddings_ids = d['encodings_document']['embeddings_ids'].to(device)
            targets = d["targets"].to(device)
            qids = d["qids"].numpy()

            outputs = model(q_embeddings, d_embeddings, q_embeddings_ids, d_embeddings_ids)

            probs = torch.sigmoid(outputs)
            preds = (probs>0.5).type(torch.int)
            loss = loss_fn(outputs.squeeze(-1), targets.float())

            targets_numpy = targets.cpu().detach().numpy()
            outputs_numpy = outputs.cpu().detach().numpy()

            combined = np.c_[ outputs_numpy, qids ] 
            combined = np.c_[ combined, targets_numpy ] 
            combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

            df = pd.concat([df, combined], ignore_index=True)

            correct_predictions += torch.sum(preds.squeeze(-1) == targets)
            losses.append(loss.item())

    map_score = calc_map(df)

    return correct_predictions.double() / n_examples, np.mean(losses), map_score


def train_drmm_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    df = pd.DataFrame(columns=['pred', 'qid', 'true'])
  
    for i, d in enumerate(data_loader):
        q_embeddings = d['encoding_query']['embeddings'].to(device)
        q_embeddings_ids = d['encoding_query']['embeddings_ids'].to(device)
        d_embeddings = d['encodings_document']['embeddings'].to(device)
        d_embeddings_ids = d['encodings_document']['embeddings_ids'].to(device)
        targets = d["targets"].to(device)
        qids = d["qids"].numpy()

        outputs = model(q_embeddings, d_embeddings, q_embeddings_ids, d_embeddings_ids)

        probs = torch.sigmoid(outputs)
        preds = (probs>0.5).type(torch.int)
        loss = loss_fn(outputs.squeeze(-1), targets.float())

        targets_numpy = targets.cpu().detach().numpy()
        outputs_numpy = outputs.cpu().detach().numpy()
        
        combined = np.c_[ outputs_numpy, qids ]
        combined = np.c_[ combined, targets_numpy ]
        combined = pd.DataFrame(combined, columns=["pred","qid", "true"])

        df = pd.concat([df, combined], ignore_index=True)

        correct_predictions += torch.sum(preds.squeeze(-1) == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i % 10 == 0: print(f"Finished Batch {i+1} - Loss {losses[-1]}")

    map_score = calc_map(df)
    
    return correct_predictions.double() / n_examples, np.mean(losses), map_score


def train_drmm(model, tokenizer, train_data_loader, val_data_loader, device, N_EPOCHS, verbose, learning_rate, batch_size, model_name, early_stopping, pre_trained_model_name, text_combination, cleaning_type):
    torch.cuda.empty_cache()
    print("Starting training") 

    model_path_dir = f'models/approach5/{model_name}'
    base_history_path = f'history/approach5/{model_name}'
    create_dir_if_not_exists(model_path_dir)
    create_dir_if_not_exists(base_history_path)
    best_model_path = model_path_dir + "/best_model.pt"
    history_path = base_history_path + "/history.pickle"

    history = defaultdict(list)
    best_map = None
    best_loss = None
    loss_not_decreasing_counter = 0
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    optimizer, scheduler = init_optimizer(model, learning_rate, train_data_loader, N_EPOCHS)

    for epoch in range(N_EPOCHS):

        print(f'Epoch {epoch + 1}/{N_EPOCHS}')
        print('-' * 20)

        train_acc, train_loss, train_map_score = train_drmm_epoch(
            model,
            train_data_loader,    
            loss_fn, 
            optimizer, 
            device, 
            scheduler, 
            len(train_data_loader.dataset.labels)
        )

        val_acc, val_loss, val_map_score = eval_drmm(
            model,
            val_data_loader,
            loss_fn, 
            device, 
            len(val_data_loader.dataset.labels)
        )    
    
        print(f'Train loss {train_loss} accuracy {train_acc} map {train_map_score}')
        print(f'Val   loss {val_loss}   accuracy {val_acc} map {val_map_score}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map_score)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map_score)

        if best_map is None or val_map_score > best_map:
            save_checkpoint(epoch, model, tokenizer, best_map, best_model_path, pre_trained_model_name, pre_trained_model_name, text_combination, cleaning_type, drop)
            best_map = val_map_score

        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            loss_not_decreasing_counter = 0
        else: loss_not_decreasing_counter += 1

        if loss_not_decreasing_counter >= early_stopping:
            save_pickle(history_path, history)
            print("Finished training")    
            return # If loss is not decreasing more after 5 epochs, return

    save_pickle(history_path, history)
    print("Finished training")   

def train_general(approach, model, tokenizer, train_data_loader, val_data_loader, device, N_EPOCHS, verbose, learning_rate, batch_size, model_name, early_stopping, pre_trained_model_name, model_type, text_combination, cleaning_type, drop):
    torch.cuda.empty_cache()
    print(f"Starting training of Approach {approach}") 

    model_path_dir = f'models/approach{approach}/{model_name}'
    base_history_path = f'history/approach{approach}/{model_name}'
    create_dir_if_not_exists(model_path_dir)
    create_dir_if_not_exists(base_history_path)
    best_model_path = model_path_dir + "/best_model.pt"
    history_path = base_history_path + "/history.pickle"

    history = defaultdict(list)
    best_map = None
    best_loss = None
    loss_not_decreasing_counter = 0
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    optimizer, scheduler = init_optimizer(model, learning_rate, train_data_loader, N_EPOCHS)

    for epoch in range(N_EPOCHS):

        print(f'Epoch {epoch + 1}/{N_EPOCHS}')
        print('-' * 20)

        train_acc, train_loss, train_map_score = train_epoch(
            approach,
            model,
            train_data_loader,    
            loss_fn, 
            optimizer, 
            device, 
            scheduler, 
            len(train_data_loader.dataset.labels)
        )

        val_acc, val_loss, val_map_score = eval_model(
            approach,
            model,
            val_data_loader,
            device, 
            len(val_data_loader.dataset.labels)
        )    
    
        print(f'Train loss {train_loss} accuracy {train_acc} map {train_map_score}')
        print(f'Val   loss {val_loss}   accuracy {val_acc} map {val_map_score}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map_score)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map_score)

        if best_map is None or val_map_score > best_map:
            save_checkpoint(epoch, model, tokenizer, best_map, best_model_path, pre_trained_model_name, model_type, text_combination, cleaning_type, drop)
            best_map = val_map_score

        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            loss_not_decreasing_counter = 0
        else: loss_not_decreasing_counter += 1

        if loss_not_decreasing_counter >= early_stopping:
            save_pickle(history_path, history)
            print("Finished training")    
            return # If loss is not decreasing more after 5 epochs, return

        save_pickle(history_path, history)
    print(f"Finished training appraoch {approach}")   
    

def train_approach6(model, tokenizer, train_data_loader, val_data_loader, device, N_EPOCHS, verbose, learning_rate, batch_size, model_name, early_stopping, pre_trained_model_name, model_type, text_combination, cleaning_type):
    torch.cuda.empty_cache()
    print("Starting training") 

    model_path_dir = f'models/approach6/{model_name}'
    base_history_path = f'history/approach6/{model_name}'
    create_dir_if_not_exists(model_path_dir)
    create_dir_if_not_exists(base_history_path)
    best_model_path = model_path_dir + "/best_model.pt"
    history_path = base_history_path + "/history.pickle"

    history = defaultdict(list)
    best_map = None
    best_loss = None
    loss_not_decreasing_counter = 0
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    optimizer, scheduler = init_optimizer(model, learning_rate, train_data_loader, N_EPOCHS)

    for epoch in range(N_EPOCHS):

        print(f'Epoch {epoch + 1}/{N_EPOCHS}')
        print('-' * 20)

        train_acc, train_loss, train_map_score = train_epoch_approach_6(
            train_data_loader,    
            loss_fn, 
            optimizer, 
            device, 
            scheduler, 
            len(train_data_loader.dataset.labels),
            model
        )

        val_acc, val_loss, val_map_score = eval_model_approach_6(
            val_data_loader,
            loss_fn, 
            device, 
            len(val_data_loader.dataset.labels),
            model
        )

        print(f'Train loss {train_loss} accuracy {train_acc} map {train_map_score}')
        print(f'Val   loss {val_loss}   accuracy {val_acc} map {val_map_score}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map_score)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map_score)

        if best_map is None or val_map_score > best_map:
            save_checkpoint(epoch, model, tokenizer, best_map, best_model_path, pre_trained_model_name, model_type, text_combination, cleaning_type, drop)
            best_map = val_map_score

        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            loss_not_decreasing_counter = 0
        else: loss_not_decreasing_counter += 1

        if loss_not_decreasing_counter >= early_stopping:
            save_pickle(history_path, history)
            print("Finished training")    
            return # If loss is not decreasing more after 5 epochs, return

    save_pickle(history_path, history)
    print("Finished training")