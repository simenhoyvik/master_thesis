import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import df_to_list, load_word2vec_model




class Approach2Dataset(Dataset):
  def __init__(self, texts, labels, qids, tokenizer, max_len):
    self.texts = texts
    self.labels = labels
    self.qids = qids
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, item):
    text = str(self.texts[item])
    label = self.labels[item]
    qids = self.qids[item]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'token_type_ids': encoding['token_type_ids'].flatten(),
      'targets': torch.tensor(label, dtype=torch.long),
      'qids' : qids
    }
  
class Approach5Dataset(Dataset):
  def __init__(self, queries, documents, labels, qids, q_len, d_len):
    self.queries = queries
    self.documents = documents
    self.labels = labels
    self.qids = qids
    self.q_len = q_len
    self.d_len = d_len
    self.w2v_model = load_word2vec_model()
  
  def __len__(self):
    return len(self.queries)
  
  def __getitem__(self, item):
    query = str(self.queries[item])
    document = str(self.documents[item])
    label = self.labels[item]
    qids = self.qids[item]

    q_tokenized = query.split()
    q_embeddings = []
    q_embeddings_ids = []

    for i in range(self.q_len):
      try:
        w = q_tokenized[i]
        q_embeddings.append(self.w2v_model[w])
        q_embeddings_ids.append(self.w2v_model.key_to_index[w])
      except:
        q_embeddings.append([0]*50)
        q_embeddings_ids.append(-1)

    d_tokenized = document.split()
    d_embeddings = []
    d_embeddings_ids = []

    for i in range(self.d_len):
      try:
        w = d_tokenized[i]
        d_embeddings.append(self.w2v_model[w])
        d_embeddings_ids.append(self.w2v_model.key_to_index[w])
      except:
        d_embeddings.append([0]*50)
        d_embeddings_ids.append(-1)

    return {
      'encoding_query' : {
        'text': query,
        'embeddings': torch.tensor(q_embeddings),
        'embeddings_ids': torch.tensor(q_embeddings_ids)
      },
      'encodings_document' : {
        'text': document,
        'embeddings': torch.tensor(d_embeddings),
        'embeddings_ids': torch.tensor(d_embeddings_ids)
      },
      'targets': torch.tensor(label, dtype=torch.long),
      'qids' : qids
    }
  
class Approach1Dataset(Dataset):
  def __init__(self, tf_idf_matrix, labels, qids):
    self.tf_idf_matrix = tf_idf_matrix
    self.labels = labels
    self.qids = qids
  
  def __len__(self):
    return self.tf_idf_matrix.shape[0]
  
  def __getitem__(self, item):
    tf_idf_matrix = self.tf_idf_matrix[item].toarray()[0]
    label = self.labels[item]
    qids = self.qids[item]

    return {
      'tf_idf_matrix': torch.tensor(tf_idf_matrix, dtype=torch.long),
      'targets': torch.tensor(label, dtype=torch.long),
      'qids' : qids
    }

class Approach3Dataset(Dataset):
  def __init__(self, queries, document_titles, document_abstracts, labels, qids, tokenizer, max_len):
    self.queries = queries
    self.document_titles = document_titles
    self.document_abstracts = document_abstracts
    self.labels = labels
    self.qids = qids
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.document_titles)
  
  def __getitem__(self, item):
    query = str(self.queries[item])
    document_title = str(self.document_titles[item])
    document_abstract = str(self.document_abstracts[item])
    label = self.labels[item]
    qids = self.qids[item]

    encoding_query = self.tokenizer.encode_plus(
      query,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    encoding_doc_title = self.tokenizer.encode_plus(
      document_title,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    encoding_doc_abstract = self.tokenizer.encode_plus(
      document_abstract,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'encoding_title' : {
        'text': query,
        'input_ids': encoding_query['input_ids'].flatten(),
        'attention_mask': encoding_query['attention_mask'].flatten(),
        'token_type_ids': encoding_query['token_type_ids'].flatten(),
      },
      'encoding_study_title' : {
        'text': document_title,
        'input_ids': encoding_doc_title['input_ids'].flatten(),
        'attention_mask': encoding_doc_title['attention_mask'].flatten(),
        'token_type_ids': encoding_doc_title['token_type_ids'].flatten(),
      },
      'encoding_study_abstract' : {
        'text': document_abstract,
        'input_ids': encoding_doc_abstract['input_ids'].flatten(),
        'attention_mask': encoding_doc_abstract['attention_mask'].flatten(),
        'token_type_ids': encoding_doc_abstract['token_type_ids'].flatten(),
      },
      'qids' : qids,
      'targets': torch.tensor(label, dtype=torch.long),
    }

def _pad_crop(item, l):
    if len(item) < l:
      item = item + [-1] * (l - len(item))
    if len(item) > l:
      item = item[:l]
    return torch.tensor(item).long().cuda()

def _mask(item, l):
  # needs padding (masked)
  if len(item) < l:
    mask = [1. for _ in item] + ([0.] * (l - len(item)))
  # no padding (possible crop)
  if len(item) >= l:
    mask = [1. for _ in item[:l]]
  return torch.tensor(mask).float().cuda()

class Approach4Dataset(Dataset):
  def __init__(self, texts, queries, documents, labels, qids, tokenizer, max_len, q_len):
    self.texts = texts
    self.queries = queries
    self.documents = documents
    self.labels = labels
    self.qids = qids
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.q_len = q_len
    self.special_tokens = 3
  
  def __len__(self):
    return len(self.documents)
  
  def __getitem__(self, item):
    text = str(self.texts[item])
    query = str(self.queries[item])
    document = str(self.documents[item])
    label = self.labels[item]
    qids = self.qids[item]

    query_tok = self.tokenizer.tokenize(query)
    document_tok = self.tokenizer.tokenize(document)

    query_tok = self.tokenizer.convert_tokens_to_ids(query_tok)
    document_tok = self.tokenizer.convert_tokens_to_ids(document_tok)

    #query_tok = [self.tokenizer.vocab[t] for t in query_tok]
    #document_tok = [self.tokenizer.vocab[t] for t in document_tok]

    doc_len = self.max_len - self.q_len - self.special_tokens

    query_mask = _mask(query_tok, self.q_len)
    document_mask = _mask(document_tok, doc_len)

    query_tok = _pad_crop(query_tok, self.q_len)
    document_tok = _pad_crop(document_tok, doc_len)

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      padding='max_length',
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )

    return {
      'query' : {
        'text': query,
        'tokens' : query_tok,
        'mask' : query_mask
      },
      'document' : {
        'text': document,
        'tokens' : document_tok,
        'mask' : document_mask
      },
      'text' : text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'token_type_ids' : encoding['token_type_ids'].flatten(),
      'qids' : qids,
      'targets': torch.tensor(label, dtype=torch.long),
    }
  
class Approach6Dataset(Dataset):
  def __init__(self, queries, document_abstracts, relevant_abstracts, labels, qids, tokenizer, max_len):
    self.queries = queries
    self.document_abstracts = document_abstracts
    self.relevant_abstracts = relevant_abstracts
    self.labels = labels
    self.qids = qids
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.document_abstracts)
  
  def encode(self, item):
    encoding = self.tokenizer.encode_plus(
      item,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      )
    return encoding
  
  def __getitem__(self, item):
    query = str(self.queries[item])
    document_abstract = str(self.document_abstracts[item])
    relevant_abstract = str(self.relevant_abstracts[item])
    label = self.labels[item]
    qids = self.qids[item]

    encoding_query = self.encode(query)
    encoding_doc_abstract = self.encode(document_abstract)
    encoding_rel_abstract = self.encode(relevant_abstract)

    return {
      'encoding_title' : {
        'text': query,
        'input_ids': encoding_query['input_ids'].flatten(),
        'attention_mask': encoding_query['attention_mask'].flatten(),
        'token_type_ids': encoding_query['token_type_ids'].flatten(),
      },
      'encoding_study_abstract' : {
        'text': document_abstract,
        'input_ids': encoding_doc_abstract['input_ids'].flatten(),
        'attention_mask': encoding_doc_abstract['attention_mask'].flatten(),
        'token_type_ids': encoding_doc_abstract['token_type_ids'].flatten(),
      },
      'encoding_relevant_document' : {
        'text': relevant_abstract,
        'input_ids': encoding_rel_abstract['input_ids'].flatten(),
        'attention_mask': encoding_rel_abstract['attention_mask'].flatten(),
        'token_type_ids': encoding_rel_abstract['token_type_ids'].flatten(),
      },
      'qids' : qids,
      'targets': torch.tensor(label, dtype=torch.long),
    }
  
class Approach62Dataset(Dataset):
  def __init__(self, texts_1, texts_2, labels, qids, tokenizer, max_len):
    self.texts_1 = texts_1
    self.texts_2 = texts_2
    self.labels = labels
    self.qids = qids
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.texts_1)
  
  def __getitem__(self, item):
    text_1 = str(self.texts_1[item])
    text_2 = str(self.texts_2[item])
    label = self.labels[item]
    qids = self.qids[item]

    encoding_1 = self.tokenizer.encode_plus(
      text_1,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    encoding_2 = self.tokenizer.encode_plus(
      text_2,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'encoding_text_1' : {
        'text': text_1,
        'input_ids': encoding_1['input_ids'].flatten(),
        'attention_mask': encoding_1['attention_mask'].flatten(),
        'token_type_ids': encoding_1['token_type_ids'].flatten(),
      },
      'encoding_text_2' : {
        'text': text_2,
        'input_ids': encoding_2['input_ids'].flatten(),
        'attention_mask': encoding_2['attention_mask'].flatten(),
        'token_type_ids': encoding_2['token_type_ids'].flatten(),
      },
      'targets': torch.tensor(label, dtype=torch.long),
      'qids' : qids
    }

def create_data_loader(df, tokenizer, max_len, batch_size, approach, text_columns, q_len, sampler, tf_idf):
  if approach == 1:
    ds = Approach1Dataset(
      tf_idf_matrix = tf_idf,
      labels = df['label'],
      qids = df['qid']
    )
  elif approach == 2:
    ds = Approach2Dataset(
      texts=df['text'].to_numpy(),
      labels=df['label'].to_numpy(),
      qids=df['qid'].to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
    )
  elif approach == 3:
    ds = Approach3Dataset(
      queries=df['title'].to_numpy(),
      document_titles=df['study_title'].to_numpy(),
      document_abstracts=df['study_abstract'].to_numpy(),
      labels=df['label'].to_numpy(),
      qids=df['qid'].to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
    )
  elif approach == 5:
    ds = Approach5Dataset(
      queries=df['title'].to_numpy(),
      documents=df['study_abstract'].to_numpy(),
      labels=df['label'].to_numpy(),
      qids=df['qid'].to_numpy(),
      q_len=50,
      d_len=512
    )
  elif approach == 6:
    ds = Approach6Dataset(
      queries=df['title'].to_numpy(),
      document_abstracts=df['study_abstract'].to_numpy(),
      relevant_abstracts=df['relevant_abstract'].to_numpy(),
      labels=df['label'].to_numpy(),
      qids=df['qid'].to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
    )
  elif approach == 62:
    ds = Approach62Dataset(
      texts_1=df['text_1'].to_numpy(),
      texts_2=df['text_1'].to_numpy(),
      labels=df['label'].to_numpy(),
      qids=df['qid'].to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
    )
  else:
    text_columns = text_columns.split()
    ds = Approach4Dataset(
      texts=df['text'].to_numpy(),
      queries=df[text_columns[0]].to_numpy(),
      documents=df[text_columns[1]].to_numpy(),
      labels=df['label'].to_numpy(),
      qids=df['qid'].to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
      q_len = q_len
    )

  return DataLoader(
    ds,
    batch_size=batch_size,
    sampler=sampler
  )