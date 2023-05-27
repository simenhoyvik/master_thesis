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
    self.freeze_bert = freeze_bert
    self.linear1 = nn.Linear(self.model.config.hidden_size, n_classes)
    self.linear2 = nn.Linear(self.model.config.hidden_size, math.floor(self.model.config.hidden_size/4))
    self.linear3 = nn.Linear(math.floor(self.model.config.hidden_size/4), math.floor(self.model.config.hidden_size/4/4))
    self.linear4 = nn.Linear(math.floor(self.model.config.hidden_size/4/4), n_classes)

    if freeze_bert:
       for p in self.model.parameters():
         p.requires_grad = False
  
  def forward(self, input_ids, attention_mask, token_type_ids):
    output = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
    )
    cls_outputs = output.pooler_output
    if not self.freeze_bert:
      output = self.linear1(cls_outputs)
    else:
      output = self.linear2(cls_outputs)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear3(output)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear4(output)

    return output

class Approach3Linear(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, model_type, freeze_bert, dropout_value):
    super(Approach3Linear, self).__init__()
    self.model1 = AutoModel.from_pretrained(pre_trained_model_name)
    self.model2 = AutoModel.from_pretrained(pre_trained_model_name)
    self.model3 = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu = nn.ReLU()
    self.freeze_bert = freeze_bert
    self.linear1 = nn.Linear(self.model1.config.hidden_size * 3, n_classes)
    self.linear2 = nn.Linear(self.model1.config.hidden_size * 3, math.floor((self.model1.config.hidden_size * 3)/4))
    self.linear3 = nn.Linear(math.floor((self.model1.config.hidden_size * 3)/4), math.floor((self.model1.config.hidden_size * 3)/4/4))
    self.linear4 = nn.Linear(math.floor((self.model1.config.hidden_size * 3)/4/4), n_classes)

    if freeze_bert:
      for p in self.model1.parameters():
        p.requires_grad = False
      for p in self.model2.parameters():
        p.requires_grad = False
      for p in self.model3.parameters():
        p.requires_grad = False
      
  
  def forward(self, input_ids_q, attention_mask_q, token_type_ids_q,
              input_ids_t, attention_mask_t, token_type_ids_t,
              input_ids_a, attention_mask_a, token_type_ids_a
    ):
    output_q = self.model1(
      input_ids=input_ids_q,
      attention_mask=attention_mask_q,
      token_type_ids=token_type_ids_q,
    )
    q_last_hidden = output_q.last_hidden_state
    q_cls_reps = q_last_hidden[:,0,:]
    output_t = self.model2(
      input_ids=input_ids_t,
      attention_mask=attention_mask_t,
      token_type_ids=token_type_ids_t,
    )
    t_last_hidden = output_t.last_hidden_state
    t_cls_reps = t_last_hidden[:,0,:]
    output_a = self.model3(
      input_ids=input_ids_a,
      attention_mask=attention_mask_a,
      token_type_ids=token_type_ids_a,
    )
    a_last_hidden = output_a.last_hidden_state
    a_cls_reps = a_last_hidden[:,0,:]

    cls_outputs = torch.cat((q_cls_reps,t_cls_reps,a_cls_reps), dim=1)

    if not self.freeze_bert:
      output = self.linear1(cls_outputs)
    else:
      output = self.linear2(cls_outputs)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear3(output)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear4(output)

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
    self.linear2 = nn.Linear(self.model1.config.hidden_size * 3, math.floor((self.model1.config.hidden_size * 3)/4))
    self.linear3 = nn.Linear(math.floor((self.model1.config.hidden_size * 3)/4), math.floor((self.model1.config.hidden_size * 3)/4/4))
    self.linear4 = nn.Linear(math.floor((self.model1.config.hidden_size * 3)/4/4), n_classes)
    self.freeze_bert = freeze_bert

    if freeze_bert:
      for p in self.model1.parameters():
        p.requires_grad = False
      for p in self.model2.parameters():
        p.requires_grad = False
      for p in self.model3.parameters():
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
    if not self.freeze_bert:
      output = self.linear1(cls_outputs)
    else:
      output = self.linear2(cls_outputs)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear3(output)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear4(output)
    return output

class Approach6_2Model(nn.Module):
  def __init__(self, n_classes, pre_trained_model_name, freeze_bert, dropout_value):
    super(Approach6_2Model, self).__init__()
    self.model1 = AutoModel.from_pretrained(pre_trained_model_name)
    self.model2 = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_value)
    self.relu = nn.ReLU()
    self.freeze_bert = freeze_bert
    self.sigmoid = nn.Sigmoid()
    self.linear1 = nn.Linear(self.model1.config.hidden_size * 2, n_classes)
    self.linear2 = nn.Linear(self.model1.config.hidden_size * 2, math.floor((self.model1.config.hidden_size * 2)/4))
    self.linear3 = nn.Linear(math.floor((self.model1.config.hidden_size * 2)/4), math.floor((self.model1.config.hidden_size * 2)/4/4))
    self.linear4 = nn.Linear(math.floor((self.model1.config.hidden_size * 2)/4/4), n_classes)    

    if freeze_bert:
      for p in self.model1.parameters():
        p.requires_grad = False
      for p in self.model2.parameters():
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
    if not self.freeze_bert:
      output = self.linear1(cls_outputs)
    else:
      output = self.linear2(cls_outputs)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear3(output)
      output = self.drop(output)
      output = self.relu(output)
      output = self.linear4(output)
    return output