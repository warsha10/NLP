#!/usr/bin/env python
# coding: utf-8

# ### Performing sentiment analysis on the Elon-musk's tweets 

# #### 1. Importing libraries

# In[41]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import spacy
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# #### 2. Importing the dataset

# In[3]:


tweet_data=pd.read_csv('Elon_musk.csv', encoding='Latin-1')
tweet_data


# In[4]:


tweet_data


# ### 3. Data preparation 
# #### 3.1 Preparing a pipeline for text preprocessing

# In[5]:


tweet_data= tweet_data.drop(labels='Unnamed: 0', axis=True)
tweet_data


# In[6]:


tweet_data.Text


# In[7]:


# remove both the leading and the trailing characters
tweet_data_modified=[Text.strip() for Text in tweet_data.Text] 


# In[8]:


tweet_data_modified


# In[9]:


# Joining the list into one string/text

tweet_data_modified=' '.join(tweet_data_modified)
tweet_data_modified


# In[10]:


# remove https or url within text
tweet_data_modified=re.sub(r'http\S+', '', tweet_data_modified)
tweet_data_modified


# In[11]:


# removing Twitter username handles from a given twitter text (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweet_data_tokenized=tknzr.tokenize(tweet_data_modified)
print(tweet_data_tokenized)


# In[12]:


corpus = []
ps=PorterStemmer()

for i in range(0,len(tweet_data_tokenized)):
    review= re.sub('[^a-zA-Z]',' ', tweet_data_tokenized[i])
    review =  re.sub(r"\b[a-zA-Z]\b", "", review)

    review=review.lower()
    review=review.split()
    review  = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' ' .join(review)
    
    
    corpus.append(review)
    


# In[13]:


corpus


# In[14]:


# removes empty strings, because they are considered in Python as False
corpus=[Text for Text in corpus if Text] 

# Again Joining the list into one string/text
cleaned_tweet_data=' '.join(corpus)


# In[15]:


cleaned_tweet_data


# ### 4. Creating a document matrix
# 
# #### 4.1 Implementing Bag of words model

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer  
cv=CountVectorizer(max_features=(2500))
X= cv.fit_transform(corpus)


# In[17]:


print(cv.vocabulary_)


# ### 5. Named Entity Recognition (NER)
# 

# In[18]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=cleaned_tweet_data
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[19]:


len(doc_block)


# In[20]:


for token in doc_block:
    print(token,token.pos_)    


# In[21]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs)


# In[23]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:50] # viewing top ten results


# In[24]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# ### 6. Emotion mining 

# In[25]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(nouns_verbs))
sentences


# In[26]:


sent_df=pd.DataFrame(tweet_data,columns=['Text'])
sent_df


# In[27]:


# Emotion Lexicon - Affin
affin=pd.read_csv('Afinn.csv',sep=',',encoding='Latin-1')
affin


# In[28]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[29]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[38]:


# manual testing
calculate_sentiment(text='great')


# In[39]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['Text'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[40]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['Text'].str.split().apply(len)
sent_df['word_count']


# In[33]:


sent_df.sort_values(by='sentiment_value')


# In[34]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[35]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[36]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[37]:


import warnings 
warnings.filterwarnings('ignore')

# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[ ]:





# In[ ]:




