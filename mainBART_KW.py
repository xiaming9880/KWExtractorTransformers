import os
import argparse
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import random
import time
import datetime
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import  Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from pytorch_pretrained_bert import BertTokenizer, BertConfig
#from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
#from transformers import AutoTokenizer, AutoModel
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
import logging
from seqeval.metrics import f1_score
import constants 




## ----------- 1- we initialize logger ---------------##

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='MainBart.log',
                    filemode='w')
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)
logger.info('message')

## ----------- 2- we retrieve command line args  ---------------##
parser = argparse.ArgumentParser(description='Keyword Extraction by Label Classification')

parser.add_argument('--training_data', type=str, default='venv/data/maui-semeval2010-train',
                    help='location of the data corpus')
parser.add_argument('--test_data', type=str, default='venv/data/maui-semeval2010-test',
                    help='location of the test data corpus')                
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--validate_data', type=str, default='venv/data/maui-semeval2010-val',
                    help='location of the valid data corpus')                
parser.add_argument('--batch_size', type=int, default=3, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=125, metavar='N',
                    help='sequence length')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='initial learning rate')
parser.add_argument('--save', type=str, default='/ia/',
                    help='path to save the final model')
parser.add_argument('--cache', type=str, default='./cache',
                    help='path to cache')
parser.add_argument('--split', type=int, default=15,
                    help='split files to 20  sentences max')



args = parser.parse_args()


## -----------    3 - Global variables initialization --------------------------
MAX_LEN = args.seq_len
bs = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
tag2idx = {constants.KEYWORD_LABEL: 0, constants.NGRAM_KEYWORD_LABEL: 1, constants.NOT_KEYWORD_LABEL: 2}
tags_vals = [constants.KEYWORD_LABEL, constants.NGRAM_KEYWORD_LABEL, constants.NOT_KEYWORD_LABEL]
max_grad_norm = 1.0
# Set the seed value all over the place to make this reproducible.
seed_val = 113
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []
#train_path = args.train_data

## -----------    4 - Global functions --------------------------

## convert function :::  lee las lineas de un fichero de texto (text_file_path)  y etiqueta las palabras en caso 
## de que sean parte de una keyword definida en el ficero key_file_path

def convert(text_file_path, key_file_path):
    sentences = ""
    for line in open(text_file_path, 'r'):
        sentences += (" " + line.rstrip())
     # sent_tokenize divide una linea en frases   
    tokens = sent_tokenize(sentences)
    key_file = open(key_file_path,'r')
    keys = [line.strip() for line in key_file]
    key_sent = []
    labels = []
    for token in tokens:
        z = [constants.NOT_KEYWORD_LABEL] * len(token.split())
        for k in keys:
            if k.lower() in token.lower():
                if len(k.split())==1:
                    try:
                        z[token.lower().split().index(k.lower().split()[0])] = constants.KEYWORD_LABEL
                    except ValueError:
                        continue
                elif len(k.split())>1:
                    try:
                        #si el indice de la palabra keyword y el de su palabra predecesora también está Ojo!!! puede haber una palabra en la frase !!
                        if token.lower().split().index(k.lower().split()[0]) and token.lower().split().index(k.lower().split()[-1]):
                            z[token.lower().split().index(k.lower().split()[0])] = constants.KEYWORD_LABEL
                            for j in range(1, len(k.split())):
                                z[token.lower().split().index(k.lower().split()[j])] = constants.NGRAM_KEYWORD_LABEL
                    except ValueError:
                        continue                       
        for m, n in enumerate(z):
            if z[m] == constants.NGRAM_KEYWORD_LABEL and z[m-1] == constants.NOT_KEYWORD_LABEL:
                z[m] = constants.NOT_KEYWORD_LABEL

        if set(z) != {constants.NOT_KEYWORD_LABEL}:
            labels.append(z)
            key_sent.append(token)
    return key_sent, labels


##  function flat_accuracy::: define como se mide la exactitud de los resultados  ( en nuestro caso es la suma de los casos de exito 
#  entre el total de resultados
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

''' 
   Returns  four  arrays  from the sentences and labels received as entry:
          tokens: [[tks1][tks2]  ....[tksn]] 
                   - Where tks1 are the tokens from sentence 1  ,
                     tks1 are the tokens from sentence 2  ... 
                     and tksn are tokens from sentence n 
          input_ids:  [[ids1][ids2]  ....[idsnsn]] 
                    - Where ids1 are ids from tokens in sentence 1 + fill items to fit length to lenght max_sentence_tokens ,
                      ids22 are  ids from tokens in sentence 2 + fill items to fit length to lenght max_sentence_tokens ... 
                      and idsnsn ids from tokens in  sentence n + fill items to fit length to lenght max_sentence_tokens
         tags:  [[tags1][tags2]  ....[tagsn]] 
                    - Where tags1 are labels assigned to tokens in sentence 1 + fill items to fit length to lenght max_sentence_tokens ,
                      tags2 are labels assigned to tokens in sentence 2 + fill items to fit length to lenght max_sentence_tokens ... 
                      and tagsn are labels assigned to tokens in  + fill items to fit length to lenght max_sentence_tokens
                                
        attention_mask: array of 0 with input_id shape
'''
def generateInputData(sentences,labels,tokenizer):
    tokens=[]
    for sent in sentences:
            tokens.append(tokenizer.tokenize(sent))
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(token) for token  in tokens] ,
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                        maxlen=MAX_LEN, value=tag2idx[constants.NOT_KEYWORD_LABEL], padding="post",
                        dtype="long", truncating="post")

    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

    return  tokens,input_ids, tags, attention_masks


class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'sentences': sample['sentences'],
                'labels': sample['labels'],
                'tokens': sample['tokens'],
                'input_ids': torch.tensor(sample['input_ids']),
                'tags': torch.tensor(sample['tags']),
                'attention_mask':torch.tensor(sample['attention_mask'])
                }




# --------- Prepare Training       
# ========================================
#              Divide  el fichero  file_path in x ficheros con max_sentences sentences in each new file
#              New file index name file_path_<index>     
# ========================================

def file_split( file_path,max_sentence=20):
    #Leemos el fichero a dividir  y lo guardamos en sentences
    f = open(file_path,'r')
    f1 = f.readlines()
    sentences = []
    for line in f1:
        sentences.append(" "+line.rstrip()+"\n")
    f.close()
    # preparamos la escritura   
    i=1
    count_lines=0
    #calcualmos el numero de ficheros salientes
    incremento = 1
    if (len(sentences)%max_sentence)==0:
        incremento = 0  
    num_files = (len(sentences)/max_sentence)+incremento
    file_names=[]
    while i< num_files:
        file_names.append(file_path+'-'+str(i)+'spl')
        f = open(file_names[i-1],'w+')
        delta = (i-1)*max_sentence
        while (count_lines < max_sentence and  (delta+count_lines)<len(sentences)):
             f.write(sentences[((i-1)*max_sentence)+count_lines])
             count_lines += 1
        count_lines = 0
        f.close()
        i += 1
    return file_names
    
def prepare_files(path, max_sentence=100):
    listFiles =os.listdir(path)
    src_files=[]
    for f in listFiles:
        if f.endswith(".txt") and not f.endswith("-justTitle.txt"):
            src_files.append(file_split(path+os.path.sep+f,max_sentence))


## --------- DataSet Definition
class FileTxtDataset(Dataset):
    def __init__(self, root_dir, tokenizer,transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.txt = sorted([f for f in os.listdir(root_dir) if f.endswith("spl")])
        self.key = sorted([f for f in os.listdir(root_dir) if f.endswith(".key")])  
        self.transform =transform

    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):
        key_file_name = self.txt[idx][0:self.txt[idx].index('.')] +constants.KEY_FILE_SUFFIX
        s, l = convert(self.root_dir + '/' +self.txt[idx], self.root_dir+'/'+key_file_name)   
        logger.info("DataLoader::getItem index : {} , fichero {} ".format(idx,self.root_dir + '/' +key_file_name ))
        tokens , input_ids,  tags,attention_mask = generateInputData(s,l,self.tokenizer)
        sample = {'sentences': s, 'labels': l, 'tokens':tokens,'input_ids':input_ids, 'tags': tags, 'attention_mask':attention_mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
        
   

class FileTxtDataset2(torch.utils.data.IterableDataset):
    def __init__(self, root_dir, tokenizer,max_batch_size,transform=None):
        super(FileTxtDataset).__init__()
        self.root_dir = root_dir
        # tokenizer to be used when split a sentence to token list
        self.tokenizer = tokenizer
         # directory where data files are  
        self.txt = sorted([f for f in os.listdir(root_dir) if not f.endswith("-justTitle.txt") and not f.endswith(".key") and not f.endswith("-CrowdCountskey") and not f.startswith('.')])
        self.key = sorted([f for f in os.listdir(root_dir) if f.endswith(".key")])  
        # transformer  is called before return to change return values
        self.transform =transform
        # index to file readen
        self.idx = random.randint(0,len(self.txt))
        # pointer to current file rteading index 
        self.local_index = 0
        # list of indices generated from divide a file in group of sentences < batch_size
        self.indices = [0]
        self.max_batch_size = max_batch_size
        # list of sentences form current file
        self.file_sentences = []
         # list of labels form current file
        self.file_labels= []

    def __iter__(self):
        result=[]
        if (self.local_index==0): 
            self.file_sentences, self.file_labels = convert(self.root_dir + '/' +self.txt[self.idx], self.root_dir+'/'+self.key[self.idx])
            # we create a list of indices where grouped sentences words are not longer than batch_size
            p_labels = 0
            for k,labels in  enumerate(self.file_labels):
                if (len(labels)+p_labels<self.max_batch_size):
                     p_labels = p_labels +len(labels)
                else:
                     p_labels=0
                     self.indices.append(k-1)               
        if (self.local_index<len(self.indices)-1):
            s=self.file_sentences[self.indices[self.local_index]:self.indices[self.local_index+1]]
            l= self.file_labels[self.indices[self.local_index]:self.indices[self.local_index+1]]
            self.local_index = self.local_index +1
        else :
            self.local_index = 0
            self.idx = random.randint(len(self.txt))
        tokens , input_ids,  tags,attention_mask = generateInputData(s,l,self.tokenizer)
        logger.info("Sentences: {}".format(s))
        sample = {'sentences': s, 'labels': l, 'tokens':tokens,'input_ids':input_ids, 'tags': tags, 'attention_mask':attention_mask}
        if self.transform:
            sample = self.transform(sample)
        return iter(sample)
    


## --------- 13. TRAINING loop

# ========================================
#               Training
# ========================================

def launch_training(training_path, training_epochs, valid_path, training_batch_size ,model, model_path ,tokenizer, optimizer):
    
    train_data = FileTxtDataset(root_dir=training_path,tokenizer=tokenizer, transform=SampleToTensor())
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=training_batch_size, num_workers=0)


    valid_data = FileTxtDataset(root_dir=valid_path, tokenizer=tokenizer,transform=SampleToTensor())
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=training_batch_size, num_workers=0)


    for epoch_index in range(0, training_epochs):
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_index + 1, training_epochs))
        logger.info('Training...')
        t0 = time.time()
        model.train()
        total_train_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # report progress
            if step % 10 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                logger.info('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                logger.info('Training...Train loss: {}'.format(total_train_loss/(nb_tr_steps+1)))  
            # add batch to gpu
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
             
            b_input_ids, b_input_mask, b_labels = batch["input_ids"][0], batch["attention_mask"][0],batch["tags"][0]
            if (b_input_mask.size()[0]<1):
                continue
            logger.info("Forward pass: inputs id shape: {}, label shape: {}, attention mask shape:{}".format( b_input_ids.size(), b_input_mask.size(), b_labels.size()))
            # Always clear any previously calculated gradients before performing a
            # backward pass
            model.zero_grad()   
            # Solve issue 
            model.model.decoder.generation_mode=False
	    # Info about forward step	
            logger.info("Forward step")	
            # forward pass
            logits = model(b_input_ids,token_type_ids=None, 
                        attention_mask=b_input_mask, labels = b_labels, return_dict=True)

            # track train loss
            total_train_loss += logits.loss.item()
            # backward pass
            logger.info("Backward step")	
            logits.loss.backward()
            
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        
            # gradient clipping  -- Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        
            # update parameters
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        logger.info("")
        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        logger.info("  Training epoch took: {:}".format(training_time))

## --------- 13. Validation loop      
# ========================================
#           Validation
# ========================================
# After the completion of each training epoch, measure our performance on
# our validation set.
        
        logger.info("")
        logger.info("Running Validation...")


        model.eval()
        t0 = time.time()
        total_eval_loss,eval_loss,  total_eval_accuracy =0, 0, 0
        nb_eval_steps, nb_eval_examples =  0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            b_input_ids = batch["input_ids"][0]
            b_input_mask = batch["attention_mask"] [0]
            b_labels = batch["tags"][0]
            if (b_input_mask.size()[0]<1):
            	continue
            with torch.no_grad():
                output= model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels,  return_dict=True)
            # Move logits and labels to CPU
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += output.loss.mean().item()
            total_eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        
        
        validation_time = format_time(time.time() - t0)
        total_eval_loss = eval_loss/nb_eval_steps
        logger.info("Validation loss: {}".format(total_eval_loss))
        logger.info("Validation Accuracy: {}".format(total_eval_accuracy/nb_eval_steps))
        logger.info(" Validation took: {:}".format(validation_time))

        logger.info("predictions shape:{}".format(len(predictions)))
        logger.info("true_labels shape:{}".format(len(true_labels)))
        torch.save(model, model_path)


    logger.info("Training/Validation  Finished. Saving Model")
    torch.save(model, model_path)
    model.save_pretrained(args.save) 

# --------- 15. Test       
# ========================================
#               Test -Evaluation Step
# ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
def launch_test(test_path,  model,tokenizer):
    
    test_data = FileTxtDataset(root_dir=test_path, tokenizer=tokenizer, transform=ToTensor())
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    model.eval()
    predictions, true_labels = [],[]
    total_eval_loss, total_eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in test_dataloader:
        b_input_ids = batch["input_ids"][0]
        b_input_mask = batch["attention_mask"] [0]
        b_labels = batch["tags"][0]
        
        with torch.no_grad():
            output = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        logits = output.logits
        logits = logits.detach().cpu().numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_test_accuracy = flat_accuracy(logits, label_ids)

        test_loss += output.loss.mean().item()
        test_accuracy += tmp_test_accuracy

        nb_test_examples += b_input_ids.size(0)
        nb_test_steps += 1

        pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
        valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
        logger.info("Test loss: {}".format(eval_loss/nb_eval_steps))
        logger.info("Test Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        logger.info("Test Predictions:{}".format(pred_tags))
        logger.info("Test F1-Score: {}".format(f1_score(pred_tags, valid_tags)))



def launch_test_without_label(test_path,  model,tokenizer):
    
    test_data = FileTxtDataset(root_dir=test_path, tokenizer=tokenizer, transform=SampleToTensor())
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    model.eval()
    test_loss, test_accuracy, nb_test_steps = 0,0,0
    predictions, true_labels = [],[]
    total_eval_loss, total_eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in test_dataloader:
        b_input_ids = batch["input_ids"]
        b_input_mask = batch["attention_mask"]
        b_labels = batch["tags"]
        
        with torch.no_grad():
            summarize = model.generate(b_input_ids[0], num_beams=4, max_length=5, early_stopping=True)
            output = model(b_input_ids[0],b_labels[0], return_dict=True)

        logits = output.logits
        
        logits = logits.detach().cpu().numpy()

        
        logger.info("********* Summarization ******************")
        summarize_phrase = [g for g in summarize]
        logger.info([tokenizer.decode(k, skip_special_tokens=True, clean_up_tokenization_spaces=True ) for k in summarize_phrase] )

        logger.info("********* SoftMax ******************")
        m =  torch.nn.Softmax(dim=1)
        probs = m(torch.from_numpy(logits[0]))
        values, predictions = probs.topk(5)
        p = list()
        for k in predictions:
            for m in k:
                p.append( m.ite/mm())
        print([tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)])

        pred_tags = p
        valid_tags = b_labels
        logger.info("predicted tags:{}".format(pred_tags))
        logger.info("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


        logger.info("********* ArgMax ******************")

        predictions = np.argmax(logits, axis=2)
        p = list()
        for k in predictions:
            for m in k:
                p.append( m )
        print([ tokenizer.decode(p , skip_special_tokens=True, clean_up_tokenization_spaces=True)] )

        pred_tags = p
        valid_tags = b_labels
        logger.info("predicted tags:{}".format(pred_tags))
        logger.info("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

        # valid_tags = [[append(l_ii)for l_ii in l_i] for l in true_labels for l_i in l ]
        logger.info("Test loss: {}".format(output.loss))



def launch_bart():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    config = BartConfig.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', num_labels=len(tags_vals))
    model_path = args.save +'bart_trained.pt'
    ## ---------12 . Optimizer -> weight regularization is  a solution to reduce the overfitting of a deep learning
    """ 
    Last keras optimization 2020 (rates from 0.01 seem to be best hyperparamater )for weight regularization for weights layers
        from keras.layers import LSTM
        from keras.regularizers import l2
    model.add(LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))) 
    Note :  BERT not include beta an gamma parametres for optimization
    """
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)
    launch_training(training_path=args.training_data,training_epochs=4,valid_path=args.validate_data,training_batch_size=1, model=model,model_path= model_path,tokenizer=tokenizer, optimizer =optimizer)
    print(model_path)
    model = BartForConditionalGeneration.from_pretrained(args.save)
    launch_test_without_label(test_path=args.test_data,model=model,tokenizer=tokenizer)

if args.split is not None :
    prepare_files(args.training_data, args.split)
    prepare_files(args.validate_data, args.split)
    prepare_files(args.test_data, args.split)
launch_bart()
