
# coding: utf-8

# In[25]:


import re
import nltk
import time
import pandas as pd
import numpy as np
import sklearn
import pickle
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from pattern.en import singularize
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
stop = stopwords.words('english')


# In[26]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')


# In[27]:


#Import Data
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#attributes=pd.read_csv('attributes.csv')
description=pd.read_csv('product_descriptions.csv')


# In[28]:


description.index=description['product_uid']
description=description.drop('product_uid',axis=1)
#Remove Nan
#attributes=attributes.dropna(how='all')
#attributes.reset_index(inplace=True)


# In[29]:


#spell correcting

from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter('\n'.join(s for s in description.as_matrix().flatten()).split())

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# In[ ]:


stop=stopwords.words('english')
def cleaning_text(sentence):
    if isinstance(sentence,basestring):
        sentence = unicode(sentence,'utf-8', errors='replace')
        sentence=sentence.lower()
        sentence=re.sub('[^\w\s]',' ', sentence) #removes punctuations
        sentence=re.sub('_', ' ', sentence) #removes punctuations
        sentence=re.sub('\d+',' ', sentence) #removes digits
        cleaned=' '.join([w for w in sentence.split() if not w in stop]) 
        cleaned=' '.join([singularize(w) for w in sentence.split() ]) #remove pural
        #removes english stopwords
        cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'NN' or pos=='JJ' or pos=='JJR' or pos=='JJS' )])
        #selecting only nouns and adjectives

        cleaned=' '.join([w for w in cleaned.split() if not len(w)<=2 ]) 
        #removes single lettered words and digits
        cleaned=cleaned.strip()
        return cleaned
    else:
        return ''


# In[30]:


def clean_text(sentence):
    if isinstance(sentence,basestring):
        sentence= unicode(sentence,'utf-8',errors='replace')
        sentence=sentence.lower()
        sentence=' '.join([stemmer.stem(w) for w in sentence.split()])
        return sentence
    else:
        return ''


# In[31]:


#Bullets_cleaned=Bullets['Bullets'].apply(lambda x: cleaning_text(x))
Description_cleaned=description['product_description'].apply(lambda x: clean_text(x))

pickle.dump(Description_cleaned, open( "Description_cleaned.p", "wb" ))
#Description_cleaned=pickle.load(open( "Description_cleaned.p", "rb" ))


# In[ ]:


Description_cleaned.index=description.index


# In[ ]:


Search_Term=pd.Series(train.search_term.append(test.search_term))
for term in Search_Term:
    term= ' '.join([correction(w) for w in term.split()])
Search_Term=Search_Term.apply(lambda x: clean_text(x))
pickle.dump(Search_Term,open('Search_Term.p','wb'))
#Search_Term=pickle.load(open('Search_Term.p','rb'))


# In[ ]:


tf_vectorizer1 = CountVectorizer(max_df=0.95, min_df=2, max_features=400,
                                stop_words='english')
tf_vectorizer2 = CountVectorizer(max_df=0.95, min_df=2, max_features=200,
                                stop_words='english')


# In[ ]:


tf1 = tf_vectorizer1.fit_transform(np.array(Description_cleaned.replace(np.nan, '')))
tf2 = tf_vectorizer2.fit_transform(np.array(Search_Term.replace(np.nan,'')))
TF = TfidfTransformer()


# In[ ]:


#tf_idf_Bullets = TF.fit_transform(tf1)
tf_idf_Description=TF.fit_transform(tf1)
tf_idf_Search_Term= TF.fit_transform(tf2)


# In[ ]:


#Trans_Bullets=pd.DataFrame(tf_idf_Bullets.toarray())
#Trans_Bullets.index=Bullets.index
Trans_Description=pd.DataFrame(tf_idf_Description.toarray())
Trans_Description.index=description.index


# In[ ]:


Trans_Search_Term=pd.DataFrame(tf_idf_Search_Term.toarray())


# In[ ]:


#Trans_Bullets.head()
Trans_Description.head()


# In[ ]:


Trans_Search_Term.head()


# In[ ]:


import xgboost as xgb


# In[ ]:


train.index=train['product_uid']
Trans_Search_Term=Trans_Search_Term[0:len(train)]
Trans_Search_Term.index=train['product_uid']
Trans_Search_Term['relevance']=train['relevance']


# In[ ]:


#train_vec=pd.merge(Trans_Bullets,Trans_Search_Term,left_index=True,right_index=True)
train_vec=pd.merge(Trans_Description,Trans_Search_Term,left_index=True,right_index=True)


# In[ ]:


Relevance=train_vec['relevance']
train_vec=train_vec.drop(['relevance'],axis=1)


# In[ ]:


param={}
param['eta']=0.01
param['max_depth']=6
param['silent']=1
param['eval_metric']='rmse'
param['min_child_weight']=3
param['subsample']=0.7
param['colsample_bytree']=0.7
num_rounds=50000


# In[ ]:


train_vec=train_vec.reset_index().drop('product_uid',axis=1)

Relevance=Relevance.reset_index().drop('product_uid',axis=1)


# In[ ]:


start_ = time.time()

#x_train = np.array(train_vec.iloc[0:50000])
#y_train = np.array(Relevance.iloc[0:50000])

#x_validation = np.array(train_vec.iloc[50000:])
#y_validation = np.array(Relevance.iloc[50000:])
x_train, x_validation, y_train, y_validation=model_selection.train_test_split(train_vec,Relevance,test_size=0.3)

xgtrain = xgb.DMatrix(x_train, label= y_train)
xgvalidation=xgb.DMatrix(x_validation,label=y_validation)


# In[ ]:


clf = xgb.train(param, xgtrain, num_rounds,evals=[ (xgtrain,'train'),(xgvalidation,'eval')],
                early_stopping_rounds=100, verbose_eval =100)


# In[ ]:


xgvalidation = xgb.DMatrix(x_validation)
y_prob = clf.predict(xgvalidation)

print 'Time elapsed: %.2f seconds' % (time.time() - start_)


# In[ ]:




