from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining
from transformers import DataCollatorForLanguageModeling, TextDatasetForNextSentencePrediction
import torch
import torch.nn as nn
from transformers import AdamW
import numpy as np
from indicnlp.tokenize import sentence_tokenize
import random
import logging as log


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class mlm_nsp_classifier(nn.Module):
    def __init__(self):
        super(mlm_nsp_classifier, self).__init__()

        self.mlm_layer1 = nn.Linear(in_features=768, out_features=768, bias=True)
        self.layer_norm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.mlm_layer2 = nn.Linear(in_features=768, out_features=127772, bias=True)
        self.nsp_layer = nn.Linear(in_features=768, out_features=2, bias=True)

    def forward(self,mlm_features, nsp_features):
        x_mlm = self.mlm_layer1(mlm_features)
        x_mlm = self.layer_norm(x_mlm)
        x_mlm = self.mlm_layer2(x_mlm)
        x_nsp = self.nsp_layer(nsp_features)
        return x_mlm, x_nsp

def read_mlm_file(file_path):
    mlm_file = open(file_path, 'r')
    data = []
    count = 0
    for line in mlm_file:
        line = line.strip()
        words = line.split()
        if len(words) <= 64:
            data.append(line)
            count = count + 1
        if count == 1000:
            break
    return data

def save_model():
    model_dict = model.state_dict()
    torch.save(model_dict, model_save_path)

def set_new_lr(rate):
    for param_group in optimizer_lm.param_groups:
        param_group['lr'] = rate
    for param_group in optimizer_mlm_nsp.param_groups:
        param_group['lr'] = rate

def schedule_learning_rate(it):
    if it <= warmup_steps:
        warmup_co = learning_rate / warmup_steps
        new_lr = (it+1) * warmup_co
        set_new_lr(new_lr)

def train(iterations, batch_size):
    mlm_current_head = 0
    nsp_head = 0
    for iteration in range(iterations):

        schedule_learning_rate(iteration)

        if mlm_current_head + batch_size <= len(mlm_data):
            mlm_batch_data = mlm_data[mlm_current_head : mlm_current_head + batch_size]
            mlm_current_head = mlm_current_head + batch_size
        else:
            mlm_batch_data = mlm_data[mlm_current_head : len(mlm_data)]
            mlm_current_head = 0

        if nsp_head + batch_size < len(nsp_data):
            data = nsp_data[nsp_head : nsp_head + batch_size]
            nsp_head = nsp_head + batch_size
        else:
            data = nsp_data[nsp_head : len(nsp_data)]
            nsp_head = 0
        length = 0
        for d in data:
            t_ids = d['input_ids'].tolist()
            if len(t_ids) > length:
                length = len(t_ids)
        batch_input_ids = np.zeros((batch_size, length), dtype = int)
        batch_attention_mask = np.zeros((batch_size, length), dtype = int)
        batch_token_type_ids = np.zeros((batch_size, length), dtype = int)
        batch_labels = np.zeros((batch_size,), dtype = int)

        for data_index, data in enumerate(data):
            input_ids = d['input_ids'].tolist()
            token_type_ids = d['token_type_ids'].tolist()
            next_sentence_label = d['next_sentence_label'].tolist()
            batch_input_ids[data_index, 0:len(input_ids)] = input_ids
            batch_attention_mask[data_index, 0:len(input_ids)] = 1
            batch_token_type_ids[data_index, 0:len(input_ids)] = token_type_ids
            batch_labels[data_index] = next_sentence_label

        nsp_input_ids = torch.tensor(batch_input_ids, dtype = torch.long)
        nsp_token_type_ids = torch.tensor(batch_token_type_ids, dtype = torch.long)
        nsp_attention_mask = torch.tensor(batch_attention_mask, dtype = torch.long)
        nsp_labels = torch.tensor( batch_labels, dtype = torch.long)

        
        mlm_inputs = tokenizer(mlm_batch_data, truncation=True, padding=True)
        mlm_labels = data_collator.mask_tokens(torch.tensor(mlm_inputs['input_ids']))[0]

        mlm_features = model(input_ids = torch.tensor(mlm_inputs['input_ids']), 
                attention_mask = torch.tensor(mlm_inputs['attention_mask']), 
                token_type_ids = torch.tensor(mlm_inputs['token_type_ids']))[0]

        nsp_features = model(input_ids = nsp_input_ids, 
                attention_mask = nsp_attention_mask, 
                token_type_ids = nsp_token_type_ids)[0][:, 0, :]

        mlm_logits, nsp_logits = mlm_nsp_classifier_model(mlm_features, nsp_features)
        
        mlm_loss = loss_fct(mlm_logits.view(-1, 127772), mlm_labels.view(-1))
        nsp_loss = loss_fct(nsp_logits.view(-1, 2), nsp_labels.view(-1))
        
        loss = mlm_loss + nsp_loss
        loss_to_print = loss.item()
        str_loss = "{0:.6f}".format(loss_to_print)
        sentence = "ITERATION "+str(iteration+1)+" : "+str(str_loss)
        log.info(sentence)

        optimizer_lm.zero_grad()
        optimizer_mlm_nsp.zero_grad()
        loss.backward()
        optimizer_lm.step()
        optimizer_mlm_nsp.step()
    save_model()

def prepare_mlm_file(file_path, output_file_path):
    mlm_file = open(file_path, 'r')
    mlm_file_write = open(output_file_path, 'w')
    for line in mlm_file:
        line = line.strip()
        if "&lt" in line or "&gt" in line:
            continue
        if len(line.split()) == 0:
            continue
        lines = sentence_tokenize.sentence_split(line, lang='hi')
        for l in lines:
            mlm_file_write.write(l.strip()+"\n")


def prepare_nsp_file(file_path, output_file_path):
    nsp_file = open(file_path, 'r')
    nsp_file_write_temp = open("temp_"+output_file_path, 'w')
    for line in nsp_file:
        line = line.strip()
        if "&lt" in line or "&gt" in line:
            continue
        if len(line.split()) == 0:
            continue
        lines = sentence_tokenize.sentence_split(line, lang='hi')
        if len(lines) == 0:
            continue
        elif len(lines) == 1:
            nsp_file_write_temp.write(lines[0].strip()+"\n\n")
        else:
            for l in lines:
                nsp_file_write_temp.write(l.strip()+"\n")
            nsp_file_write_temp.write("\n")
    nsp_file_write_temp.close()
    nsp_file_write_temp = open("temp_"+output_file_path, 'r')
    count = 0
    for l in nsp_file_write_temp:
        count = count + 1
    count_2 = 0
    nsp_file_write_temp.close()
    nsp_file_write_temp = open("temp_"+output_file_path, 'r')
    nsp_file_write = open(output_file_path, 'w')
    for l in nsp_file_write_temp:
        if count_2 <= count - 2:
            count_2 = count_2 + 1
            nsp_file_write.write(l)
        else:
            break

if __name__ == "__main__" :

    wiki_file = "wiki-ne.txt"
    mlm_output_file_path = 'mlm_file_ne.txt'
    nsp_output_file_path = 'nsp_file_ne.txt'
    log_file = 'log_mlm_nsp.txt' 
    bert_model = 'extened_bert/'
    iterations = 10
    batch_size = 2
    learning_rate = 0.00005
    warmup_steps = 8
    model_save_path = "saved_models/pretrained_mbert.pt"

    log.basicConfig(format='%(asctime)s %(message)s', 
            datefmt='%m/%d/%Y %I:%M:%S %p', 
            level = log.INFO,
            filename = log_file,
            filemode = "w")

    prepare_mlm_file(wiki_file, mlm_output_file_path)
    prepare_nsp_file(wiki_file, nsp_output_file_path)
    mlm_data = read_mlm_file(mlm_output_file_path)

    
    loss_fct = nn.CrossEntropyLoss() #loss for mlm

    mlm_nsp_classifier_model = mlm_nsp_classifier() # langauge models extra part 

    tokenizer = BertTokenizer.from_pretrained(bert_model) #tokenizer for the model
    model = BertModel.from_pretrained(bert_model) #model for mlm and extra objective
    base_model = BertModel.from_pretrained(bert_model) #for regularization

    model.train()
    base_model.eval()
    optimizer_lm = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), eps=1e-08, lr = 1.) #optimizer for language model 
    optimizer_mlm_nsp = torch.optim.Adam(mlm_nsp_classifier_model.parameters(),betas=(0.9, 0.98), eps=1e-08, lr = 1.) #optimizer for extra mlm classidier

    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = True) #data collator for masking
    nsp_data = TextDatasetForNextSentencePrediction(tokenizer, file_path = nsp_output_file_path, block_size = 256, nsp_probability = 0.5)
    
    train(iterations, batch_size)