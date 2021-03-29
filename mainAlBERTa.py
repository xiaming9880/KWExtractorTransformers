import os
import  constants
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
from transformers import AlbertTokenizer, AlbertConfig, AlbertForTokenClassification
#from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
import logging
import preprocessFiles
import statisticsLocal




## ----------- 1- we initialize logger ---------------##

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='MainAlbertTrash.log',
                    filemode='w')
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s:%(asctime)s - %(levelname)-8s %(message)s')
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

parser.add_argument('--training_data', type=str,
                    help='location of the data corpus')
parser.add_argument('--test_data', type=str, default='venv/data/science',
                    help='location of the test data corpus')                
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--validate_data', type=str, default='venv/data/science',
                    help='location of the valid data corpus')                
parser.add_argument('--batch_size', type=int, default=3, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=125, metavar='N',
                    help='sequence length')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='initial learning rate')
#parser.add_argument('--save', type=str, 
parser.add_argument('--save', type=str, default='venv/models/alberta_full',
#parser.add_argument('--save', type=str, default='venv/models/alberta03507',
                   help='path to save the final model')
parser.add_argument('--cache', type=str, default='./cache',
                    help='path to cache')

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

def getKeysFromFile(key_file_path):
    key_file = open(key_file_path,'r')
    keys = [line.strip() for line in key_file]
    return keys
    
## convert function :::  lee las lineas de un fichero de texto (text_file_path)  y etiqueta las palabras en caso 
## de que sean parte de una keyword definida en el ficero key_file_path
def convert(text_file_path, key_file_path,tokenizer):
    sentences = ""
    for line in open(text_file_path, 'r'):
        if (len(line.strip())>0):
            sentences += (" " +line.strip())
     # sent_tokenize divide una linea en frases   
    tokens = sent_tokenize(sentences)
    key_file = open(key_file_path,'r')
    keys = [line.strip() for line in key_file]
    key_sent = []
    labels = []
    for sentence in tokens:
        token_sentence = tokenizer.tokenize(sentence)
        tokens_array = np.array(token_sentence)
        z = [constants.NOT_KEYWORD_LABEL] * len(token_sentence)
        for k in keys:
            key_tokenized= tokenizer.tokenize(k)
            for k in key_tokenized:
                item_index = np.where(tokens_array == k)[0]
                for i in item_index:
                    z[i] = constants.KEYWORD_LABEL                         
        labels.append(z)
        key_sent.append(token_sentence)
    return key_sent, labels, keys


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
def generateInputData(tokens,labels,tokenizer):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(token) for token  in tokens] ,
                            maxlen=MAX_LEN, dtype="long", truncating="pre", padding="pre")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                        maxlen=MAX_LEN, value=tag2idx[constants.NOT_KEYWORD_LABEL], padding="pre",
                        dtype="long", truncating="pre")

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


## --------- DataSet Definition
class FileTxtDataset(Dataset):
    def __init__(self, root_dir, tokenizer,transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.txt = sorted([f for f in os.listdir(root_dir) if f.endswith(".txt")])
        self.spl = sorted([f for f in os.listdir(root_dir) if f.endswith("spl")])
        self.key = sorted([f for f in os.listdir(root_dir) if f.endswith(".key")])  
        self.transform =transform
        self.current_file=""
        self.idx=0

    def __len__(self):
        return len(self.spl)
    
    def getCurrentTxtFile( self ):
      return self.current_file

    def __getitem__(self, idx):
        self.idx= idx
        print("__getitem__",idx)
        if (self.getCurrentTxtFile() != self.spl[idx][0:self.spl[idx].index('.')]+constants.TXT_FILE_SUFFIX):
            self.current_file = self.spl[idx][0:self.spl[idx].index('.')]+constants.TXT_FILE_SUFFIX
        key_file_name = self.spl[idx][0:self.spl[idx].index('.')] +constants.KEY_FILE_SUFFIX
        s, l , keys= convert(self.root_dir + '/' +self.spl[idx], self.root_dir+'/'+key_file_name,self.tokenizer)   
        logger.info("DataLoader::getItem index : {} , fichero {} ".format(idx,self.root_dir + '/' +self.spl[idx] ))
        tokens , input_ids,  tags,attention_mask = generateInputData(s,l,self.tokenizer)
        sample = {'sentences': s, 'labels': l, 'tokens':tokens,'input_ids':input_ids, 'tags': tags, 'attention_mask':attention_mask}
        if self.transform:
            sample = self.transform(sample)
        #logger.info('analyzed data = {0} '.format(s))
        return sample
        
    def getFiles(self):
        return self.txt

    def getCurrentSplFile(self):
        return self.spl[self.idx]


# ------------------------------------------------
# ------------------------------------------------
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
        sample = {'sentences': s, 'labels': l, 'tokens':tokens,'input_ids':input_ids, 'tags': tags, 'attention_mask':attention_mask}
        if self.transform:
            sample = self.transform(sample)
        return iter(sample)
    


## --------- 13. TRAINING loop

# ========================================
#               Training
# ========================================

def launch_training(training_path, training_epochs, valid_path, training_batch_size ,model, model_path ,tokenizer, optimizer):
    try:
        train_data = FileTxtDataset(root_dir=training_path,tokenizer=tokenizer, transform=SampleToTensor())
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=training_batch_size)


        valid_data = FileTxtDataset(root_dir=valid_path, tokenizer=tokenizer,transform=SampleToTensor())
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=training_batch_size)


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
                #model.model.decoder.generation_mode=False
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
    except:
        if (model is not None):
            model.save_pretrained(args.save) 
            torch.save(model, model_path)


# --------- 15. Test       
# ========================================
#               Test -Evaluation Step
# ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
def launch_test(test_path,  model,tokenizer):
    
     
    test_data = FileTxtDataset(root_dir=test_path, tokenizer=tokenizer, transform=SampleToTensor())
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    model.eval()
    predictions, true_labels = [],[]
    total_eval_loss, total_eval_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    test_loss, test_accuracy= 0,0
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
    p = list()
    for k in predictions:
            for m in k:
                p.append( m )
    print([ tokenizer.decode(p , skip_special_tokens=True, clean_up_tokenization_spaces=True)] )
    print('valid tags:',valid_tags)
    logger.info("Test loss: {}".format(test_loss/nb_test_steps))
    logger.info("Test Accuracy: {}".format(test_accuracy/nb_test_steps))
    logger.info("Test Predictions:{}".format(pred_tags))
    logger.info("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))




def launch_test_without_label(test_path,  model,tokenizer):
    
    test_data = FileTxtDataset(root_dir=test_path, tokenizer=tokenizer, transform=SampleToTensor())
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    model.eval()
    test_loss, test_accuracy, nb_test_steps = 0,0,0
    predictions =[]
    total_eval_loss, total_eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    p = list()
    pred_tags = list()
    true_labels= list()
    for batch in test_dataloader:

        b_input_ids = batch["input_ids"][0]
        b_input_mask = batch["attention_mask"][0]
        b_labels =batch["tags"][0]
        
        
        with torch.no_grad():
            predictions=list()
            output= model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels,  return_dict=True)
            # Move logits and labels to CPU
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend( p.tolist()for p in np.argmax(logits, axis=2))
            #predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            keys_ids = list()
            for i,i_val in enumerate(predictions):
                for j, j_val in enumerate(i_val):
                    if ((i_val[j] ==1) or (i_val[j] ==0)):
                        keys_ids.append(b_input_ids[i][j].item())
            
             
            kw_token= [tokenizer.decode(key_id,skip_special_tokens=True) for key_id in keys_ids]
            #kw_token= tokenizer.decode(keys_ids,skip_special_tokens=True).split()
    
            # tratamos sufijos antes de hacer el strip
            # kw_token=preprocessFiles.suffixReplacement(test_path+'/'+test_data.getCurrentFile(),kw_token, '<')
            kw_token =  list(dict.fromkeys([token.strip() for token in kw_token]))
            # eliminamos stopword
            kw_token=preprocessFiles.removeStopWords(kw_token)
            # buscamos ngrams
            kw_token=list(dict.fromkeys(kw_token))
            kw_token.extend(preprocessFiles.check_compound_keyword(test_path+'/'+test_data.getCurrentSplFile(),kw_token))
            print('keyword:',j_val,kw_token)
            p.extend(kw for kw in kw_token)             
    #eliminamos duplicados
    #pred_tags =  list(dict.fromkeys(p))
    pred_tags =  list(dict.fromkeys([ preprocessFiles.nlp(key)[:-1].text +' '+ preprocessFiles.nlp(key)[-1].lemma_  if len(key.split())>1 else preprocessFiles.nlp(key)[0].lemma_ for key in p] ))
    valid_tags = b_labels
    files = test_data.getFiles()  
    logger.info("***************************************" )
    
    key_file_list = sorted([f for f in os.listdir(test_path) if f.endswith( constants.KEY_FILE_SUFFIX)]) 
    for file in key_file_list:
        true_labels.extend(getKeysFromFile(test_path+os.path.sep+file))
    true_labels =  list(dict.fromkeys([ preprocessFiles.nlp(key)[:-1].text +' '+ preprocessFiles.nlp(key)[-1].lemma_  if len(key.split())>1 else preprocessFiles.nlp(key)[0].lemma_ for key in true_labels] ))
    logger.info("predicted tags:{}".format(pred_tags))
    logger.info("valid tags:{}".format(true_labels))
    logger.info("***************************************" )
    logger.info("Number of predicted keys:{}".format(len(pred_tags)))
    logger.info("Number of Valid keys:{}".format(len(true_labels)))
    logger.info("true positive:{}".format(statisticsLocal.tp(pred_tags,true_labels)))
    logger.info("false negative:{}".format(statisticsLocal.fn(pred_tags,true_labels)))
    logger.info("false positivos:{}".format(statisticsLocal.fp(pred_tags,true_labels)))
    logger.info("F1-Score: {}".format(statisticsLocal.f1_score(statisticsLocal.tp(pred_tags,true_labels),statisticsLocal.fn(pred_tags,true_labels),statisticsLocal.fp(pred_tags,true_labels))))



def launch_test_directory(test_path,  model,tokenizer):
    
    test_data = FileTxtDataset(root_dir=test_path, tokenizer=tokenizer, transform=SampleToTensor())
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

   
    nb_test_steps  = 0
    item_list = list()
    predictions =[]
    # Dictionary  List { pred_tags, true_labels, txt_file}
    item = None
    current_batch_file = None
    model.eval()
    for batch in test_dataloader:
        # When we are exracting keywords from a new txt file then we save metrics values
        if ((current_batch_file is None) | (test_data.getCurrentTxtFile()!=current_batch_file) ):
            if current_batch_file is not None:
                item['pred_tags'] =  list(dict.fromkeys([ preprocessFiles.nlp(key)[:-1].text +' '+ preprocessFiles.nlp(key)[-1].lemma_  if len(key.split())>1 else preprocessFiles.nlp(key)[0].lemma_ for key in item['pred_tags']] ))
                logger.info("*********Evaluation for file: {} *********************".format(current_batch_file))
                logger.info("predicted tags:{}".format(item['pred_tags']))
                item['true_labels'].extend(getKeysFromFile(test_path+os.path.sep+item['file_batch'][0:item['file_batch'].index('.')] +constants.KEY_FILE_SUFFIX))
                #true_labels =  list(dict.fromkeys([word.lemma_ for key in true_labels for word in preprocessFiles.nlp(key)] )) 
                item['true_labels'] =  list(dict.fromkeys([ preprocessFiles.nlp(key)[:-1].text +' '+ preprocessFiles.nlp(key)[-1].lemma_  if len(key.split())>1 else preprocessFiles.nlp(key)[0].lemma_ for key in item['true_labels'] ] ))
                tp = statisticsLocal.tp(item['pred_tags'],item['true_labels'])
                fn = statisticsLocal.fn(item['pred_tags'],item['true_labels'])
                fp = statisticsLocal.fp(item['pred_tags'],item['true_labels'])
                item['recall'] =statisticsLocal.recall(tp,fp,fn)
                item['fscore'] =statisticsLocal.f1_score(tp,fp,fn)
                item['precision']=statisticsLocal.precision(tp,fp,fn)
                logger.info("valid tags:{}".format(item['true_labels']))
                logger.info("Number of predicted keys:{}".format(len(item['pred_tags'])))
                logger.info("Number of Valid keys:{}".format(len(item['true_labels'])))
                logger.info("true positive:{}".format(tp))
                logger.info("false negative:{}".format(fn))
                logger.info("false positivos:{}".format(fp))
                logger.info("F1-Score: {}".format(item['fscore']))
                logger.info("Recall: {}".format(item['recall']))
                logger.info("Precision: {}".format(item['precision']))
                item_list.append(item)
            current_batch_file = test_data.getCurrentTxtFile()
            item = { 'pred_tags':list(),'true_labels':list(),'file_batch':current_batch_file,'f_score':float,'recall:':float,'precision':float } 
        
        b_input_ids = batch["input_ids"][0]
        b_input_mask = batch["attention_mask"][0]
        b_labels =batch["tags"][0]
    
        with torch.no_grad():
            predictions=list()
            output= model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels,  return_dict=True)
            # Move logits and labels to CPU
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
 
            
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

            keys_ids = list()
            for i,i_val in enumerate(predictions):
                for j, j_val in enumerate(i_val):
                    if ((i_val[j].item() ==1) or (i_val[j].item()==0)):
                        keys_ids.append(b_input_ids[i][j].item())


        
            kw_token=tokenizer.convert_ids_to_tokens(keys_ids,skip_special_tokens=True)
            kw_token =  list(dict.fromkeys( [ k.replace('▁','')for k in kw_token]))
            # delete stopword
            kw_token=preprocessFiles.removeStopWords(kw_token)
            #comprobamos que los tokens están presnetes en el archivo
            # we work in  suffix
            kw_token=preprocessFiles.suffixReplacement(test_path+'/'+test_data.getCurrentSplFile(),kw_token, '_')
            # retrieve ngrams
            kw_token=list(dict.fromkeys(kw_token))
            kw_token.extend(preprocessFiles.check_compound_keyword2(test_path+'/'+test_data.getCurrentSplFile(),kw_token))
            print('keyword:',j_val.item(),kw_token)
            item['pred_tags'].extend(kw for kw in kw_token)    
            nb_test_steps = nb_test_steps+1     
   
    
    
    item['pred_tags'] =  list(dict.fromkeys([ preprocessFiles.nlp(key)[:-1].text +' '+ preprocessFiles.nlp(key)[-1].lemma_  if len(key.split())>1 else preprocessFiles.nlp(key)[0].lemma_ for key in item['pred_tags']] ))
    logger.info("*********Evaluation for file: {} *********************".format(current_batch_file))
    logger.info("predicted tags:{}".format(item['pred_tags']))
    item['true_labels'].extend(getKeysFromFile(test_path+os.path.sep+item['file_batch'][0:item['file_batch'].index('.')] +constants.KEY_FILE_SUFFIX))
    #true_labels =  list(dict.fromkeys([word.lemma_ for key in true_labels for word in preprocessFiles.nlp(key)] )) 
    item['true_labels'] =  list(dict.fromkeys([ preprocessFiles.nlp(key)[:-1].text +' '+ preprocessFiles.nlp(key)[-1].lemma_  if len(key.split())>1 else preprocessFiles.nlp(key)[0].lemma_ for key in item['true_labels'] ] ))
    tp = statisticsLocal.tp(item['pred_tags'],item['true_labels'])
    fn = statisticsLocal.fn(item['pred_tags'],item['true_labels'])
    fp = statisticsLocal.fp(item['pred_tags'],item['true_labels'])
    item['recall'] =statisticsLocal.recall(tp,fp,fn)
    item['fscore'] =statisticsLocal.f1_score(tp,fp,fn)
    item['precision']=statisticsLocal.precision(tp,fp,fn)
    logger.info("valid tags:{}".format(item['true_labels']))
    logger.info("Number of predicted keys:{}".format(len(item['pred_tags'])))
    logger.info("Number of Valid keys:{}".format(len(item['true_labels'])))
    logger.info("true positive:{}".format(tp))
    logger.info("false negative:{}".format(fn))
    logger.info("false positivos:{}".format(fp))
    logger.info("F1-Score: {}".format(item['fscore']))
    logger.info("Recall: {}".format(item['recall']))
    logger.info("Precision: {}".format(item['precision']))
    item_list.append(item)
    logger.info("*************** Final Result *************")

    f1_score , recall, precision = 0.0, 0.0, 0.0
    for el in item_list:
        f1_score = f1_score+ el['fscore']
        recall = recall + el['recall']
        precision = precision + el['precision']
    logger.info("F1-Score Average: {}".format(f1_score/len(item_list)))
    logger.info("Recall: {}".format(recall/len(item_list)))
    logger.info("Precision: {}".format(precision/len(item_list)))




def launch(training_flag, test_flag): 
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    if training_flag:
        model  =  AlbertForTokenClassification.from_pretrained('albert-base-v2', num_labels=len(tags_vals))
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
        launch_training(training_path=args.training_data,training_epochs=args.epochs,valid_path=args.validate_data,training_batch_size=1, model=model,model_path= model_path,tokenizer=tokenizer, optimizer =optimizer)
    if test_flag: 
        if args.save:
            model_path = args.save +'pytorch_model.bin'
            config = AlbertConfig.from_json_file(args.save +'/config.json')
            model = AlbertForTokenClassification.from_pretrained(args.save, config=config)
        else:
              model  =  AlbertForTokenClassification.from_pretrained('albert-base-v2', num_labels=len(tags_vals))
        launch_test_directory(test_path=test_flag,model=model,tokenizer=tokenizer)
launch(None,'/Users/irenecid/Desktop/tesis/machine-learning/languageRepresentation/bert/venv/data/Documentos de prueba')


