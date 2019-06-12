
# coding: utf-8

# In[1]:


import csv

import pandas as pd
import sklearn_crfsuite
import os
import numpy as np
import time
# In[27]:



def strip_iob(iob_tag):
    tag = iob_tag.replace("B-", "")
    tag = tag.replace("I-", "")
    return tag


# In[4]:


def is_new_tag(prev, current):
    if "O" in prev:
        prev_t, prev_w = "O", "O"
    else:
        prev_t, prev_w = prev.split("-")
    if "O" in current:
        current_t, current_w = "O", "O"
    else:
        current_t, current_w = current.split("-")

    if prev_w != current_w:
        return True
    else:
        if prev_t =="B" and current_t =="I":
            return False
        elif prev_t =="I" and current_t =="B":
            return True
        elif prev_t =="I" and current_t =="I":
            return False
        else:
            return False

def returnSentences(folder,nlp):
    sentences = []
    for i in os.listdir(os.path.join(folder)):
        lemma = []
        pos = []
        ner = []
        shape =[]
        alpha =[]
        stop =[]
        df = pd.read_csv(os.path.join( folder,i),sep="\t",quoting=csv.QUOTE_NONE)
        df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        for doc in nlp.pipe(df['0'].astype('unicode').values, batch_size=150,n_threads=-1):
            if doc.is_parsed:
                lemma.append([n.lemma_ for n in doc])
                pos.append([n.pos_ for n in doc])
                ner.append([n.tag_ for n in doc])
                shape.append([n.shape_ for n in doc])
                alpha.append([n.is_alpha for n in doc])
                stop.append([n.is_stop for n in doc])

            else:
                # We want to make sure that the lists of parsed results have the
                # same number of entries of the original Dataframe, so add some blanks in case the parse fails
                tokens.append(None)
                lemma.append(None)
                pos.append(None)
                ner.append(None)
                shape.append(None)
                alpha.append(None)
                stop.append(None)
#         print(pos)        
        df['lemma'] = lemma
        df['pos'] = pos
        df['ner'] = ner
        df['shape'] = shape
        df['alpha'] = alpha
        df['stop'] = stop

        data =  list(zip(*[df[c].values.tolist() for c in ['0','pos','lemma','ner','shape','alpha','stop']]))
        sentences.append(data)
        
    return sentences

# In[12]:

def check_n_makedirs(path):
    if not os.path.exists(path):
        print_info("Creating folder: {}".format(path))
        os.makedirs(path)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
        
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1][0]
    lemma = sent[i][2][0]
    ner = sent[i][3][0]
    shape = sent[i][4][0]
    alpha = sent[i][5][0]
    stop = sent[i][6][0]
    
    features = {
        'bias': 1.0,
#         'word.lower()': word.lower() if type(word) is str else False,
#         'word[-2:]': word[-2:] if type(word) is str else word,
#         'word[:2]': word[-2:] if type(word) is str else word,
        'word[-4:]': word[-4:] if type(word) is str else word,
        'word[:4]': word[-4:] if type(word) is str else word,
        'word.isupper()': word.isupper() if type(word) is str else False ,
        'word.isalnum()':word.isalnum() if type(word) is str else False ,
        'word.istitle()': word.istitle() if type(word) is str else False,
        'word.isdigit()': word.isdigit() if type(word) is str else False,
#         'word.ishash': "#" in word if type(word) is str else False,
        'word.hasnumber()':hasNumbers(word) if type(word) is str else False,
        'word.isdot': "." in  word if type(word) is str else False,
        'word.uppercharcount': sum(1 for c in word if c.isupper()) if type(word) is str else 0,
        'word.lowercharcount': sum(1 for c in word if c.islower()) if type(word) is str else 0,
        'word.digitcount':sum(1 for c in word if c.isdigit()) if type(word) is str else 0,
        'word.length':len(word) if type(word) is str else 0,
        'word.vocablength':len(set(word)) if type(word) is str else -1,
#         'word.isroad': "road" in word.lower() if type(word) is str else False,
#         'word.hasslash': "/" in word if type(word) is str else False,
#         'word.hashyphen': ":" in word if type(word) is str else False,
#         'word.currency':word in ["pound","$","£","dollar"] if type(word) is str else False,
        'lemma':lemma,
        'postag': postag,
        'postag[:2]': postag[:2],
        'nertag': ner,
        'shape':shape,
        'alpha':alpha,
        'stop':stop,
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1][0]
        nertag1 = sent[i-1][2][0]
        lemma1 = sent[i-1][2][0]
        ner1 = sent[i-1][3][0]
        shape1 = sent[i-1][4][0]
        alpha1 = sent[i-1][5][0]
        stop1 = sent[i-1][6][0]
        features.update({
#             '-1:word[-2:]': word1[-2:] if type(word) is str else word,
#             '-1:word[:2]': word1[-2:] if type(word) is str else word,
#             '-1:word.lower()': word1.lower() if type(word1) is str else False,
            '-1:word.istitle()': word1.istitle() if type(word1) is str else False,
            '-1:word.isupper()': word1.isupper() if type(word1) is str else False,
            '-1:word.isdigit()': word1.isdigit() if type(word1) is str else False,
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:lemma':lemma1,
            '-1:nertag': nertag1,
            '-1:shape':shape1,
            '-1:alpha':alpha1,
            '-1:stop':stop1
            
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1][0]
        nertag1 = sent[i+1][2][0]
        lemma1 = sent[i+1][2][0]
        ner1 = sent[i+1][3][0]
        shape1 = sent[i+1][4][0]
        alpha1 = sent[i+1][5][0]
        stop1 = sent[i+1][6][0]
        
        features.update({
#             '+1:word.lower()': word1.lower() if type(word1) is str else False,
            '+1:word.istitle()': word1.istitle() if type(word1) is str else False,
            '+1:word.isupper()': word1.isupper() if type(word1) is str else False,
            '+1:word.isdigit()': word1.isdigit() if type(word1) is str else False,
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:lemma':lemma1,
            '+1:nertag': nertag1,
            '+1:shape':shape1,
            '+1:alpha':alpha1,
            '+1:stop':stop1,
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,pos,lemma,ner,shape,alpha,stop, label in sent]

def sent2tokens(sent):
    return [token for token,pos,lemma,ner,shape,alpha,stop, label in sent]

# In[13]:


# In[5]:
def process(nlp):

	full_path = "/home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/preprocessed_data/" #environ['DEMO_DATA_PATH']
	full_path


	# In[6]:




	# In[7]:


	#Basically in the test folder we have to keep all the csv files with two rows
	start = time.time()
	test_sents = returnSentences("/home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/preprocessed_data/test",nlp)
	print(time.time()-start)

	# In[8]:




	# In[10]:


	prediction_folder ="/home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/clientx_data_iterator/predictions"


	# In[11]:


	os.system('rm -rf /home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/clientx_data_iterator/predictions')
	os.system('mkdir -p /home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/clientx_data_iterator/predictions')



	start1 = time.time()
	X_test = [sent2features(s) for s in test_sents]
	# y_test = [sent2labels(s) for s in test_sents]
	print(time.time()-start1)

	# In[ ]:

	start2 = time.time()
	from sklearn.externals import joblib 
	# Load the model from the file
	crf = ""
	if crf == "": 
	    crf = joblib.load('playground/clientx/CRFModelExpt2.pkl')

	print(time.time()-start2)
	# In[15]:

	start3 = time.time()
	# Use the loaded model to make predictions 
	predictions = crf.predict(X_test)
	print(time.time()-start3)

	# In[16]:




	# In[17]:


	OUT_DIR ="/home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/clientx_data_iterator/"
	_prediction_col="predictions"
	# Get the files from test folder and zip it with predictions
	for each_prediction, file in zip(predictions, os.listdir(full_path+"/test/")):

	    df = pd.read_csv(os.path.join(full_path,"test",file), sep="\t", quoting=csv.QUOTE_NONE)
	    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

	    df[_prediction_col] = [j for i, j in
	                          zip(df["0"].astype(str).values.tolist(), each_prediction)]
	    
	    out_dir = os.path.join(OUT_DIR, "predictions")
	    check_n_makedirs(out_dir)
	    #df.to_csv(os.path.join(out_dir, os.path.basename(file)), index=False)

	print(df)
	doc_text=""
	enter=False
	for index, row in df.iterrows():
	    if row["predictions"] != "O":
	        if index == 0 or not enter:
	            text = row["0"]
	            prev_tag = row["predictions"]
	            enter = True

	        else:
	            # second index onwards
	            if is_new_tag(prev_tag, row["predictions"]):
	                doc_text = doc_text + text + "~" + strip_iob(prev_tag)+"\n"
	                text = row["0"]

	            else:
	                text = text + " " + row["0"]
	            prev_tag = row["predictions"]
	doc_text = doc_text + text + "~" + strip_iob(prev_tag) + "\n"

	print(doc_text)

	with open("/home/sampathm/Projects/imaginea/vitaFlow/clientx_dataset/clientx_data_iterator/predictions/OUTFILE.csv", "w") as post_file:
	    post_file.write("Item~Tag\n")
	    post_file.write(doc_text)
