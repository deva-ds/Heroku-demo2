#!/usr/bin/env python
# coding: utf-8

# ## Deployment of ML models in Heroku using FLASK -2

# In[1]:


import pandas as np


# In[14]:


df=pd.read_csv('spam.csv',encoding="latin-1")


# In[4]:


df


# In[15]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True,axis=1)


# In[16]:


df


# In[8]:


df.info()


# In[12]:


df['class']=pd.get_dummies(df['class'],drop_first=True).astype(int)


# In[13]:


df


# In[17]:


df['class']=df['class'].map({'ham':0,'spam':1})


# In[18]:


df


# In[20]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[28]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[29]:


corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[30]:


len(corpus)


# In[36]:


import gensim
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess


# In[37]:


words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))


# In[38]:


len(words)


# In[43]:


corpus_filtered = []
y_filtered = []

for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    
    if review:  # Only keep non-empty reviews
        corpus_filtered.append(review)
        y_filtered.append(df['class'][i])


# In[46]:


len(corpus_filtered),len(y_filtered)


# In[47]:


type(corpus_filtered),type(y_filtered)


# In[49]:


y= pd.Series(y_filtered)


# In[51]:


y.head()


# In[52]:


## Lets train Word2vec from scratch
model=gensim.models.Word2Vec(words)


# In[53]:


def avg_word2vec(doc):
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)


# In[54]:


from tqdm import tqdm


# In[55]:


#apply for the entire sentences
import numpy as np
X=[]
for i in tqdm(range(len(words))):
    X.append(avg_word2vec(words[i]))


# In[56]:


len(X),type(X)


# In[57]:


##independent Features
X_new=np.array(X,dtype=object)


# In[59]:


pd.Series(X)


# In[60]:


## this is the final independent features
df_list = []

for i in range(len(X_new)):
    df_list.append(pd.DataFrame(X_new[i].reshape(1, -1)))

df1 = pd.concat(df_list, ignore_index=True)


# In[61]:


df1.head()


# In[62]:


df1.isnull().sum()


# In[63]:


df1['Output']=y


# In[64]:


df1.head()


# In[66]:


df1.dropna(inplace=True)


# In[67]:


df1.head()


# In[68]:


df1.isnull().sum()


# In[70]:


## Independent Feature
X=df1.drop('Output',axis=1)


# In[113]:


X


# In[72]:


y=df1['Output']


# In[114]:


y


# In[103]:


## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[104]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()


# In[105]:


classifier.fit(X_train,y_train)


# In[106]:


y_pred=classifier.predict(X_test)


# In[107]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))


# In[108]:


print(classification_report(y_test,y_pred))


# In[109]:


model = RandomForestClassifier()


# In[110]:


model.fit(X,y)


# In[111]:


import pickle


# In[112]:


with open('model.pkl','wb') as file:
    pickle.dump(model,file)
print("Model saved as model.pkl")

with open('model.pkl','rb') as file:
    load_model=pickle.load(file)

# Test the loaded model
print(
    load_model.predict([[2,9,6]])
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')


# In[44]:


dataset


# In[45]:


dataset.columns


# In[46]:


dataset.isnull().sum()


# In[47]:


dataset.info()


# In[48]:


median_value = dataset['test_score(out of 10)'].median()
dataset['test_score(out of 10)'].fillna(median_value, inplace=True)


# In[49]:


dataset.isnull().sum()


# In[50]:


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]


# In[51]:


word_to_num ={'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}


# In[52]:





# In[53]:


dataset['experience'] = dataset['experience'].map(word_to_num)


# In[54]:


dataset


# In[59]:


df=dataset.copy()


# In[60]:


df


# In[61]:


median_val = df['experience'].mode()
df['experience'].fillna(median_val, inplace=True)


# In[62]:


df


# In[63]:


median_val = dataset['experience'].mode()
dataset['experience'].fillna(median_val, inplace=True)


# In[64]:


X=dataset.iloc[:,:3]


# In[65]:


X


# In[23]:


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


# In[68]:


X.info()


# In[66]:


y=df['salary($)']


# In[67]:


y


# In[69]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[72]:


#Fitting model with trainig data
model = regressor.fit(X, y)


# In[ ]:





# In[73]:


# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")


# In[76]:


with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Test the loaded model
print(
    loaded_model.predict([[2,9,6]])
    )


# In[77]:


get_ipython().system('pip freeze > requirements.txt')


# In[78]:


get_ipython().system('pip install pipreqs')


# In[80]:


get_ipython().system('pipreqs . --force')


# In[81]:


import pkg_resources
import sys

# Get all imported modules
imports = {mod.key for mod in pkg_resources.working_set}
with open("requirements.txt", "w") as f:
    for mod in imports:
        f.write(mod + "\n")


# In[83]:


get_ipython().system('jupyter nbconvert --to script model.ipynb')


# In[ ]:




