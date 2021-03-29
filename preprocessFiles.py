
import os
import numpy as np
import random
import time
import datetime
import en_core_web_sm
import spacy
from spacy.lemmatizer import Lemmatizer, NOUN
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token
import torch
from spacy.lang.en import English
from abc import ABCMeta, abstractmethod
from spacy.matcher import Matcher
import nltk
import re


nlp = en_core_web_sm.load()
list_of_rules = [
    ["ADJ", "NOUN"],
    ["ADJ","ADJ", "NOUN"],
    ["ADJ", "NOUN","ADP","NOUN"],
    ["ADJ","ADJ", "NOUN","ADP","NOUN"],
    ["ADJ", "NOUN","ADP","ADJ","NOUN"],
    ["NOUN", "ADP", "NOUN"],
    ["NOUN","ADP","ADJ","NOUN"],
    # more rules here...
]
rules = [[{"POS": i} for i in j] for j in list_of_rules]
matcher = Matcher(nlp.vocab)
matcher.add("rules", None, *rules)

nltk.download('punkt')
sentence_marks=[".",";","!",";","¡", "?", "¿","(",")","-",]

class TextFilter(object):
    @abstractmethod
    def __filter__(self,text):
       raise NotImplementedError

class ChapterFilter(TextFilter):
    def __init__(self,expression='((^\d+).([A-Z|\s|:]+)$)'):
        self.chapterRE=re.compile(expression, re.MULTILINE)
        self.textChapter = list()

    #crea una lista de capitulos segun la expresión regular
    def __filter__(self,text):
        it = self.chapterRE.finditer(text)
        idxChapter = 0
        chapterEnd = 0
        for match in it:
            self.textChapter.append({ 'numChapter': idxChapter+1,'title':match.group(3).strip(),'chapterStart':match.start(),'chapterTextStart':match.end(),'chapterEnd':0})
            if (idxChapter>0):
                self.textChapter.__getitem__(idxChapter-1)['chapterEnd']=match.start()-1         
            idxChapter+=1
        if (len(self.textChapter)>0):
               self.textChapter.__getitem__(idxChapter-1)['chapterEnd']=len(text)
        ##for chapter in self.textChapter:
           # print('chapterTitle:', chapter['title'], '\t start:',chapter['chapterStart'], '\t end:',chapter['chapterEnd'])
        return self.textChapter
        
    def removeChaptersByName(self,text,chapters):
        if (len(self.textChapter)==0):
            self.__filter__(text)
        for chapter in reversed(self.textChapter):
            if (chapter['title'] in chapters):
                text=text[0:chapter['chapterStart']]+text[chapter['chapterEnd']+1:]
        return text
   
# Receives a list of tokens and return a list of them in wich every token from the list that is not 
# sentence mark list or whic lexem  is condidered as stop word list in nlp 
def removeStopWords(tokens):
    filtered_sentence =list()
    for word in tokens:
        if len(word)>2 and ( word not in sentence_marks):
            lexeme = nlp.vocab[word]
            if (lexeme.is_stop == False) and (lexeme.is_punct == False) :
                filtered_sentence.append(word) 
    return filtered_sentence








# Returns a list with the a items' stem
def stem(a, p = nltk.PorterStemmer()):
    r = []
    lemmatizer = nlp.vocab.morphology.lemmatizer
    for word in a:
        
        #r.append(p.stem(word[0].lower()))
        r.append(lemmatizer(word.lower(), NOUN)[0].lower())
        
    return r

# returns the words in the text that matchs the keys:
# example-> suffixReplacement(file, keys, "##") 
#               keys[1]= ##macros  -> 
#               file = " with macroscopic properties ....macrosphere...."
#           returns  [macroscopic , macrospehere]

def suffixReplacement(file_path, keys,suffixChain ):
    suffix = list()
    for candidate in keys:
        if candidate.strip().startswith(suffixChain,0,len(suffixChain)):
            item = candidate.strip()[len(suffixChain):]
            if item not in suffix:
                suffix.append(item)
    if (len(suffix)>0):
        f = open(file_path,'r')
        f1 = f.readlines()
        current_sentence = ""
        for line in f1:
            line = line.strip('\n')
            for suf in suffix:
                if (line.find(suf)>0):
                    tokens  = nlp(line)
                    for token in tokens:
                        if (token.text.find(suf)>0) and (token.text.find(suf) +len(suf)==len(token)) and (token.text in keys):
                            keys.append(token.text)
        for suf in suffix:
            item = suffixChain+suf
            while item in keys:
                keys.remove(item)
        f.close()
    return keys



def checkExistsInFile(file_path, keys):
    f = open(file_path,'r')
    f1 = f.readlines()
    current_sentence = ""
    existkeys=list()
    f.close()
    for candidate in keys:
        found = False
        for line in f1:
            line = line.strip('\n')
            if (not found) and (line.find(candidate)>0):
                found = True
                tokens = nlp(line)
                for token in tokens:
                    if (token.text == candidate):
                            existkeys.append(token.text)
   
    return existkeys
# returns a list of tokens with stem in the list of keywords
# example :  check_keys_in_txt(chunk, keys)   
#       keys  -> [property]
#       chunk -> [ " a list of properties"]
# returns : "properties"

def check_keys_in_txt(chunk, keys):
    items = chunk.text.split()
    items_lemma = stem(items)
    keys_lemma = stem(keys)
    compound_key=''
    for i,item in enumerate(items_lemma):
        lexeme = nlp.vocab[item]
        if lexeme.is_stop:
            if len(compound_key)>0:
                compound_key=compound_key +' '+items[i].lower()
        elif item.lower() in keys_lemma:
            compound_key=compound_key +' '+items[i].lower()
            keys_lemma.remove(item.lower())
    return compound_key


# returns the ngrams 
# example-> check_compound_keyword(file, keys) 
#               keys[1]= [" macroscopic", "properties"] 
#               file = " with macroscopic properties observable by electrical microspcopics...."
#           returns -> ["macroscopic", "properties", "macroscopic properties"] 
def check_compound_keyword(file_path, keys):
    sentences = list()
    f = open(file_path,'r')
    f1 = f.read()
    sentences = nltk.sent_tokenize(f1) 
    ngram = []
    for i, s in  enumerate(sentences):
        doc = nlp(s.lower())
        matches = matcher(doc)
        #dividimos a frase en composiciones de nombres
        keys_to_remove=[]
        for match_id, start, end in matches:
             chunk =  doc[start:end]
             print('chunk:',chunk)
             print('keys:',keys)
             for j,k in enumerate(keys):
                #comprobamos si los chunks están formados por keys 
                compound_keys = check_keys_in_txt(chunk, keys)
                if (len(compound_keys.strip())>=len(chunk.text)):
                     if compound_keys.strip() not in ngram:
                         ngram.append(compound_keys.strip()) 
                         keys_to_remove.append(k)
                #forma parte de una agrupacion nominal no compuesta
                else:
                    if (k in keys_to_remove):
                        keys_to_remove.remove(k)
        for j,k in enumerate(keys_to_remove): 
            if (k in keys):
                keys.remove(k)
    print('ngram' ,ngram) 
    return ngram

def check_compound_keyword2(file_path, keys):   
    sentences = list()
    f = open(file_path,'r')
    f1 = f.read()
    sentences = nltk.sent_tokenize(f1) 
    ngram = []
    f.close()
        # ya tenemos las lineas divididas en frases
    for i, s in  enumerate(sentences):
        doc = nlp(s)
        #dividimos a frase en composiciones de nombres
        for chunk in doc.noun_chunks:
            for j,k in enumerate(keys):
                #comprobamos si las 
                compound_keys = check_keys_in_txt(chunk, keys)
                if (len(compound_keys.strip())>len(chunk.root.text)):
                    if compound_keys.strip() not in ngram:
                        ngram.append(compound_keys.strip())
    print('ngram' ,ngram)
    return ngram

def calculate_porcentage_keys_phrases(file_path,key_file_path):
    sentences = file_to_sentence_punkt(file_path)
    nb_words = 0
    matcher = PhraseMatcher(nlp.vocab,attr='LOWER')
    kf = open(key_file_path)
    keys= kf.readlines()
    patterns = [nlp.make_doc(k.strip()) for k in keys]
    matcher.add("KeywordList", None, *patterns) 
    kw_sentences_index=[]
    # guardamos dos listas con los indices de las frases que contienen palabras claves y  de las que no
    for i, s in  enumerate(sentences):
        doc = nlp(s)
        nb_words = nb_words +len(s.split())
        matches = matcher(doc)
        if (len(matches)>0):
          kw_sentences_index.append(i)
    print("numero  de  palabras:",nb_words)
    print("numero de frases :",len(sentences)) 
    return (len(kw_sentences_index)/len(sentences)),nb_words,len(sentences),len(keys)

def file_split_by_prob_distribution( sentences,file_path,key_file_path,max_sentence=20, delta=0.5 ):

    #Leemos las keywords y las cargamos como expresiones regulares
    matcher = PhraseMatcher(nlp.vocab,attr='LOWER')
    kf = open(key_file_path)
    keys= kf.readlines()
    patterns = [nlp.make_doc(k.strip()) for k in keys]
    matcher.add("KeywordList", None, *patterns) 
    kw_sentences_index=[]
    nkw_sentences_index=[]

    # guardamos dos listas con los indices de las frases que contienen palabras claves y  de las que no
    for i, s in  enumerate(sentences):
        doc = nlp(s)
        matches = matcher(doc)
        if (len(matches)>0):
          kw_sentences_index.append(i)
        else:
          nkw_sentences_index.append(i)
   
    # T = len(sentences)
    # K = len(kw_entence)
    # delta = proporción entre  0 y 1 
    # tal que min : 0 
    #     -> solo las frases con  palabras claves, 
    # tal que max : 1
    #      -> todas las frases
    # (T-K)*delta + K = 0 
    # (T-K)*delta es el numero de frases que se añadiran para cumplir la proporción
    
    s_delta= len(sentences)*delta
    kw_w= round(abs(s_delta-len(kw_sentences_index)) )
       
    w = torch.empty(1, len(sentences))
    dist = torch.nn.init.zeros_(w)
    # ponemos a 1 todos los indices de la matriz que corresponden con palabras claves
    dist[0][np.asarray(kw_sentences_index)]=1
    print('kw_sentences_index:',kw_sentences_index)
    #obtenemos una dsitribución normal con valores medio en el 1 y los valores proximos contiguos con mayor valor
    print('dist  ',dist)
    #dist=dist.normal_(mean=0.5,std=0.5)
    #print('dist normal ',dist)
    # reemplazmos valores pro debajo de la media +- 0.2 por cero
    #dist  = dist.masked_fill_(dist.le(0.6), 0.)
    #dist  = dist.masked_fill_(dist.ge(1.4), 0.)
    #print('dist ceros ',dist)
    #dist.masked_fill_(dist.ne (0), 1)
    
    if (torch.sum(dist)<len(sentences)*delta):
        mask= torch.zeros(len(nkw_sentences_index))
        rand_idx=torch.randperm(len(nkw_sentences_index))
        index=rand_idx[torch.arange(round(kw_w))]
        mask[index]=1
        tmp_index = torch.tensor(nkw_sentences_index,  dtype=torch.int)
        result = mask*tmp_index
        dist[0][np.asarray(result)]=1
    elif (torch.sum(dist)>len(sentences)*delta):
        mask= torch.ones(len(kw_sentences_index))
        rand_idx=torch.randperm(len(kw_sentences_index))
        index=rand_idx[torch.arange(round(kw_w))]
        mask[index]=0
        tmp_index = torch.tensor(kw_sentences_index,  dtype=torch.int)
        result = mask*tmp_index
        dist = torch.nn.init.zeros_(w)
        dist[0][np.asarray(result)]=1

    print('dist finale ',dist)
     
    final_sentences =[]
    for i,value in enumerate(dist[0].tolist()):
        if (value ==  1.):
            final_sentences.append(sentences[i])
    # preparamos la escritura   
    i=1
    count_lines=0
    #calculamos el numero de ficheros salientes
    incremento = 1
    if (len(final_sentences)%max_sentence)==0:
        incremento = 0  
    num_files = (len(final_sentences)/max_sentence)+incremento
    file_names=[]
    while i< num_files:
        file_names.append(file_path+'-'+str(i)+'spl')
        f = open(file_names[i-1],'w+')
        delta = (i-1)*max_sentence
        while (count_lines < max_sentence and  (delta+count_lines)<len(final_sentences)):
             f.write(final_sentences[((i-1)*max_sentence)+count_lines]+"\n")
             count_lines += 1
        count_lines = 0
        f.close()
        i += 1
    return file_names

def file_split( sentences ,key_file_path ,  max_sentence=20):   
  
    # preparamos la escritura   
    i=1
    count_lines=0
    #calcualmos el numero de ficheros salientes
    incremento = 1
    if (len(sentences)%max_sentence)==0:
        incremento = 0  
    num_files = (len(sentences)/max_sentence)+incremento
    file_names=[]
    while (i< num_files):
        file_names.append(file_path+'-'+str(i)+'spl')
        f = open(file_names[i-1],'w+')
        delta = (i-1)*max_sentence
        while (count_lines < max_sentence and  (delta+count_lines)<len(sentences)):
             f.write(sentences[((i-1)*max_sentence)+count_lines]+".\n")
             count_lines += 1
        count_lines = 0
        f.close()
        i += 1
    return file_names


# returns a file tokenized by sentences using punkt sentence tokenizer
def  file_to_sentence_punkt(file_path):
    filter=ChapterFilter()
    f = open(file_path)
    text = filter.removeChaptersByName(f.read(),['REFERENCES'])
    sentences = nltk.sent_tokenize(text) 
    #for s1 in sentences:
    #    print("***"+s1+'\n')
    f.close()
    
    return sentences

        
# returns a file tokenized by sentences searching "." as sentence separator
def  file_to_sentence_endSentence(file_path):
    sentences  =  []
    kw_sentences_index = []
    nkw_sentences_index = [] 
    sentences  =  []
    f = open(file_path,'r')
    f1 = f.readlines()
    current_sentence = ""
    for line in f1:
        line = line.strip('\n')
        start_line_cursor=0 
        while(line.find(".")>=0):
            aux=line[start_line_cursor:].strip()
            current_sentence+=line[start_line_cursor:line.index(".")]
            #actualizamos el cursor apuntando a la subcadena donde empiez la si
            start_line_cursor=line.find(".")+1
            # en aux guardamos lo que queda de línea, 
            line = line[line.index('.')+1:]
            if (len(aux) >0 and aux[0].isupper()): 
                if (len(current_sentence.strip())>0):
                    sentences.append(current_sentence)
                    current_sentence ="" 
            elif(len(aux) >0):
                    current_sentence += line                
        current_sentence +=line
    if (len(current_sentence.strip())>0):
        sentences.append(current_sentence)
    f.close()
    return sentences


def file_split( file_path, key_file_path ,  max_sentence=20):
    #Leemos el fichero  y lo guardamos en la lista de sentences 
    sentences  =  []
    f = open(file_path,'r')
    f1 = f.readlines()
    current_sentence = ""
    for line in f1:
        line = line.strip('\n')
        start_line_cursor=0 
        while(line.find(".")>=0):
            aux=line[line.find("."):].strip()
            current_sentence+=line[start_line_cursor:line.index(".")]
            start_line_cursor=line.find(".")+1
            line = line[line.index('.')+1:]
            if (len(aux) >0 and aux[0].isupper()): 
                if (len(current_sentence.strip())>0):
                    sentences.append(current_sentence)
                current_sentence =" "  
        current_sentence +=" "+ line
    
    if (len(current_sentence.strip())>0):
        sentences.append(current_sentence)
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
             f.write(sentences[((i-1)*max_sentence)+count_lines]+".\n")
             count_lines += 1
        count_lines = 0
        f.close()
        i += 1
    return file_names



def prepare_files(path, min, max):
    listFiles =os.listdir(path)
    listFiles.sort()
    src_files=[]
    file_path=None
    key_path=None
    random.seed( datetime.datetime.now())
    for f in listFiles:
        if f.endswith(".txt") and not f.endswith("-justTitle.txt"):
            file_path = path+os.path.sep+f
        if f.endswith(".key"):
            key_path = path+os.path.sep+f
        if ((file_path != None) and  (key_path != None)):
            x = random.random()
            x_norm = (x*(max-min))+min
            print ('x_norm = ',x_norm)
            src_files.append(file_split_by_prob_distribution(file_to_sentence_punkt(file_path),file_path,key_path ,20,x_norm))
            file_path  = None
            key_path  = None


def split_files(path):
    listFiles =os.listdir(path)
    listFiles.sort()
    src_files=[]
    file_path=None
    key_path=None
    for f in listFiles:
        if f.endswith(".txt") and not f.endswith("-justTitle.txt"):
            file_path = path+os.path.sep+f
        if f.endswith(".key"):
            key_path = path+os.path.sep+f
        if ((file_path != None) and  (key_path != None)):
            src_files.append(file_split(file_path,key_path ,100))
            file_path  = None
            key_path  = None


#prepare_files('/Users/irenecid/Desktop/tesis/machine-learning/languageRepresentation/bert/venv/data/Documentos de prueba',1.0,1.0)
#('venv/data/maui-semeval2010-train/Chemistry-2S-ch4.txt')
#keys = list()
#keys.append('international')
#keys.append('system')
#keys.append('units')
#check_compound_keyword('/Users/irenecid/Desktop/tesis/machine-learning/languageRepresentation/bert/venv/data/science/energy&matter.txt-5spl',keys)
#"/Users⁩/irenecid⁩/Downloads⁩/⁨SemEval2010-Maui (1)⁩"
#print("Porcentaje de frases con keywords ->", calculate_porcentage_keys_phrases('/Users/irenecid/Desktop/tesis/machine-learning/languageRepresentation/bert/venv/data/science/energy&matter.txt','/Users/irenecid/Desktop/tesis/machine-learning/languageRepresentation/bert/venv/data/science/energy&matter.key'))


"""
txt_files = sorted([f for f in os.listdir('/Users/irenecid/Downloads/Inspec/docsutf8') if f.endswith("txt")])
key_files = sorted([f for f in os.listdir('/Users/irenecid/Downloads/Inspec/keys') if f.endswith(".key")])
indice = 0
porcentaje_acumulado = 0
nwords_acumulado = 0
nsentences_acumulado=0
nkeys_acumulado=0
porcentaje=0
for txt_file, key_file in zip(txt_files,key_files):
   print("****************************" )
   print("File :",txt_file )
   print("indice :",indice )
   indice = indice +1
   porcentaje , nwords, nsentences, keys= calculate_porcentage_keys_phrases('/Users/irenecid/Downloads/Inspec/docsutf8/'+txt_file,'/Users/irenecid/Downloads/Inspec/keys/'+key_file)
   porcentaje_acumulado =porcentaje_acumulado +porcentaje
   nwords_acumulado=nwords+nwords_acumulado
   nsentences_acumulado=nsentences_acumulado+nsentences
   nkeys_acumulado=nkeys_acumulado+keys
   print("Porcentaje de frases con keywords ->", porcentaje)
   print("Media: ->", porcentaje_acumulado/indice)
   print("Media words ->", nwords_acumulado)
   print("Media sentences ->", nsentences_acumulado)
   print("Media keys ->", nkeys_acumulado)
"""
       