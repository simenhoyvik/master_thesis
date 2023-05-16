import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, AlbertModel, AutoModel, AutoConfig

class Approach1Model(nn.Module):
  def __init__(self, n_classes, tf_idf_size, dropout_value):
    super(Approach1Model, self).__init__()
    if dropout_value != None: self.drop = nn.Dropout(p=dropout_value)
    else: self.drop = None
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.linear1 = nn.Linear(tf_idf_size, n_classes)
  
  def forward(self, inputs):
    if self.drop != None:
      output = self.linear1(self.drop(inputs))
    else:
      output = self.linear1(inputs)
    return output


class BinaryClassifier(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, model_type, freeze_bert, dropout_value):
    super(BinaryClassifier, self).__init__()
    self.model = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(self.model.config.hidden_size, n_classes)
    #self.linear1 = nn.Linear(self.model.config.hidden_size, math.floor(self.model.config.hidden_size/4))
    self.linear2 = nn.Linear(math.floor(self.model.config.hidden_size/4), 16)
    self.linear3 = nn.Linear(16, n_classes)

    if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False
  
  def forward(self, input_ids, attention_mask, token_type_ids):
    output = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
    )
    pooler_output = output.pooler_output
    output = self.drop(pooler_output)
    output = self.relu(output)
    output = self.linear1(output)
    return output

class Approach3Linear(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, model_type, freeze_bert, dropout_value):
    super(Approach3Linear, self).__init__()
    self.model = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(self.model.config.hidden_size * 3, n_classes)
    #self.linear1 = nn.Linear(self.model.config.hidden_size * 3, math.floor(self.model.config.hidden_size*3/4))
    self.linear2 = nn.Linear(math.floor(self.model.config.hidden_size*3/4), 16)
    self.linear3 = nn.Linear(16, n_classes)

    if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False
  
  def forward(self, input_ids_q, attention_mask_q, token_type_ids_q,
              input_ids_t, attention_mask_t, token_type_ids_t,
              input_ids_a, attention_mask_a, token_type_ids_a
    ):
    output_q = self.model(
      input_ids=input_ids_q,
      attention_mask=attention_mask_q,
      token_type_ids=token_type_ids_q,
    )
    q_last_hidden = output_q.last_hidden_state
    q_cls_reps = q_last_hidden[:,0,:]
    output_t = self.model(
      input_ids=input_ids_t,
      attention_mask=attention_mask_t,
      token_type_ids=token_type_ids_t,
    )
    t_last_hidden = output_t.last_hidden_state
    t_cls_reps = t_last_hidden[:,0,:]
    output_a = self.model(
      input_ids=input_ids_a,
      attention_mask=attention_mask_a,
      token_type_ids=token_type_ids_a,
    )
    a_last_hidden = output_a.last_hidden_state
    a_cls_reps = a_last_hidden[:,0,:]

    cls_outputs = torch.cat((q_cls_reps,t_cls_reps,a_cls_reps), dim=1)
    output = self.drop(cls_outputs)
    output = self.relu(output)
    output = self.linear1(output)
    #output = self.linear1(outputs)
    #output = self.relu(output)
    #output = self.linear2(output)
    #output = self.relu(output)
    #output = self.linear3(output)
    return output


class Approach6Model(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, freeze_bert, dropout_value):
    super(Approach6Model, self).__init__()
    self.model1 = AutoModel.from_pretrained(pre_trained_model_name)
    self.model2 = AutoModel.from_pretrained(pre_trained_model_name)
    self.model3 = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(self.model1.config.hidden_size * 3, n_classes)

    if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False
  
  def forward(self, input_ids_q, attention_mask_q, token_type_ids_q,
              input_ids_a, attention_mask_a, token_type_ids_a,
              input_ids_r, attention_mask_r, token_type_ids_r
    ):
    output_title = self.model1(
      input_ids=input_ids_q,
      attention_mask=attention_mask_q,
      token_type_ids=token_type_ids_q,
    )
    pooled_output_title = output_title.pooler_output
    output_study_abstract = self.model2(
      input_ids=input_ids_a,
      attention_mask=attention_mask_a,
      token_type_ids=token_type_ids_a,
    )
    pooled_output_study_abstract = output_study_abstract.pooler_output
    output_relevant_document = self.model3(
      input_ids=input_ids_r,
      attention_mask=attention_mask_r,
      token_type_ids=token_type_ids_r,
    )
    pooled_output_relevant_document = output_relevant_document.pooler_output

    cls_outputs = torch.cat((pooled_output_title, pooled_output_study_abstract, pooled_output_relevant_document), dim=1)
    output = self.drop(cls_outputs)
    output = self.relu(output)
    output = self.linear1(output)
    return output

class Approach6_2Model(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, freeze_bert, dropout_value):
    super(Approach6_2Model, self).__init__()
    self.model1 = AutoModel.from_pretrained(pre_trained_model_name)
    self.model2 = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.linear1 = nn.Linear(self.model1.config.hidden_size * 2, n_classes)

    if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False
  
  def forward(self, input_ids_1, attention_mask_1, token_type_ids_1,
              input_ids_2, attention_mask_2, token_type_ids_2,
    ):
    output_1 = self.model1(
      input_ids=input_ids_1,
      attention_mask=attention_mask_1,
      token_type_ids=token_type_ids_1,
    )
    pooler_output_1 = output_1.pooler_output
    output_2 = self.model2(
      input_ids=input_ids_2,
      attention_mask=attention_mask_2,
      token_type_ids=token_type_ids_2,
    )
    pooler_output_2 = output_2.pooler_output
    cls_outputs = torch.cat((pooler_output_1, pooler_output_2), dim=1)
    output = self.drop(cls_outputs)
    output = self.relu(output)
    output = self.linear1(output)
    return output

class Approach5Pairwise(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, model_type, freeze_bert, dropout_value):
    super(Approach5Pairwise, self).__init__()
    self.model = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(self.model.config.hidden_size, n_classes)

    if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False
  
  def forward(self, input_ids_pos, attention_mask_pos, token_type_ids_pos,
              input_ids_neg, attention_mask_neg, token_type_ids_neg
              ):
    output_pos = self.model(
      input_ids=input_ids_pos,
      attention_mask=attention_mask_pos,
      token_type_ids=token_type_ids_pos,
    )
    pooler_output_pos = output_pos.pooler_output
    logits_pos = self.linear1(pooler_output_pos)

    output_neg = self.model(
      input_ids=input_ids_neg,
      attention_mask=attention_mask_neg,
      token_type_ids=token_type_ids_neg,
    )
    pooler_output_neg = output_neg.pooler_output
    logits_neg = self.linear1(pooler_output_neg)

    return logits_pos, logits_neg

class CustomBertModel(nn.Module):
    def __init__(self, config, n_classes, pre_trained_model_name, model_type, freeze_bert):
      super(CustomBertModel, self).__init__()
      if model_type == 'bert': self.model = BertModel.from_pretrained(pre_trained_model_name, config=config)
      elif model_type == 'albert': self.model = AlbertModel.from_pretrained(pre_trained_model_name, config=config)
      elif model_type == 'roberta': self.model = RobertaModel.from_pretrained(pre_trained_model_name, config=config)
      else: self.model = AutoModel.from_pretrained(pre_trained_model_name)

      if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        embedding_output = self.model.embeddings(input_ids, token_type_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.model.encoder(embedding_output, extended_attention_mask, output_hidden_states=True)
        encoded_layers = [layer for layer in  encoded_layers.hidden_states]

        return [embedding_output] + encoded_layers
    

class DRMM(nn.Module):
    def __init__(self, dim_term_gating=50, n_bins=11, q_len=50):
        super(DRMM, self).__init__()
        self.simmat_model = DrmmSimmat()
        self.hist_model = DrmmHistogram(bins=11)
        
        self.linear1 = nn.Linear(n_bins * q_len, 1)
        self.gating = nn.Linear(dim_term_gating * q_len, 1, bias=False)
        
    def forward(self, q_embeddings, d_embeddings, q_embeddings_ids, d_embeddings_ids):
        sim_matrix = self.simmat_model(q_embeddings, d_embeddings, q_embeddings_ids, d_embeddings_ids)
        histogram = self.hist_model(sim_matrix, q_embeddings_ids, d_embeddings_ids)

        BATCH, QLEN, BINS = histogram.shape
        histogram = histogram.reshape(BATCH, QLEN * BINS)

        out_ffn = self.linear1(histogram)
        out_tgn = F.softmax(self.gating(q_embeddings.reshape(BATCH, QLEN * q_embeddings.shape[1])).squeeze())

        matching_score = torch.sum(out_ffn * out_tgn,dim=1)
        return matching_score


class CedrDrmmRanker(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, max_len, tokenizer, config, model_type, freeze_bert, dropout_value):
    super(CedrDrmmRanker, self).__init__()
    NBINS = 11
    HIDDEN = 1
    self.n_classes = n_classes
    self.max_len = max_len
    self.special_tokens = 3 # = [CLS] and 2x[SEP]
    self.tokenizer = tokenizer
    self.freeze_bert = freeze_bert
    self.model = CustomBertModel(config, self.n_classes, pre_trained_model_name, model_type, freeze_bert)
    self.hidden_size = self.model.model.config.hidden_size
    self.channels = self.model.model.config.num_hidden_layers + 2
    self.simmat = SimmatModule()
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(p=dropout_value)
    self.histogram = DRMMLogCountHistogram(NBINS)
    self.linear1 = torch.nn.Linear(NBINS * self.channels + self.hidden_size, self.n_classes)
    #self.linear2 = torch.nn.Linear(HIDDEN, self.n_classes)

  def forward_model(self, input_ids, attention_mask, token_type_ids, query_tok, doc_tok, query_mask, document_mask):
    batch_size, q_len = query_tok.shape
    max_doc_token_len = self.max_len - q_len - self.special_tokens

    result = self.model(input_ids, token_type_ids, attention_mask)

    # extract relevant subsequences for query and doc
    query_results = [r[:batch_size, 1:q_len+1] for r in result]
    doc_results = [r[:, q_len+2:-1] for r in result]
    doc_results = [un_subbatch(r, doc_tok, max_doc_token_len) for r in doc_results]

    # build CLS representation
    cls_results = []
    for layer in result:
      cls_output = layer[:, 0]
      cls_result = []
      for i in range(cls_output.shape[0] // batch_size):
        cls_result.append(cls_output[i*batch_size:(i+1)*batch_size])
      cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
      cls_results.append(cls_result)

    return cls_results, query_results, doc_results

  def forward(self, input_ids, attention_mask, token_type_ids, query_tok, doc_tok, query_mask, document_mask):
    cls_reps, query_reps, doc_reps = self.forward_model(input_ids, attention_mask, token_type_ids, query_tok, doc_tok, query_mask, document_mask)
    simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
    histogram = self.histogram(simmat, doc_tok, query_tok)
    BATCH, CHANNELS, QLEN, BINS = histogram.shape
    histogram = histogram.permute(0, 2, 3, 1)
    output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
    # repeat cls representation for each query token
    cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
    cls_output = torch.cat([output, cls_rep], dim=1)
    output = self.drop(cls_output)
    output = self.relu(output)
    output = self.linear1(output)
    output = output.reshape(BATCH, QLEN, self.n_classes)
    return output.sum(dim=1)