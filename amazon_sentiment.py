#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'jupyternotify')
import warnings
warnings.filterwarnings('ignore')


# # Txt Sentiment Analysis of Amazon Customer Reviews of Pet Supplies

# In[2]:


#import libraries used in analysis
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import regex as re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from sklearn.model_selection import train_test_split

#import libraries necessary for text representation
import gensim
from gensim.models import Word2Vec, KeyedVectors
#import keras packages for building a neural network
import keras
from keras.models import Sequential
from keras.models import Model
from keras import layers
from keras.layers import GlobalMaxPooling1D, Dropout, Dense, Conv1D, Dense, Input, LSTM, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
#Import h5py for saving weights to file for future use
import h5py


# In[3]:


#set pandas print options to visualize full output of print and display statements
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[4]:


# Set the random seed to ensure repeatability in the process
seed = 88


# # Data Aquisition
# ## Read json format data set into a Pandas Dataframe

# In[5]:


get_ipython().run_cell_magic('time', '', "%%notify\n#Read the data contained in the Amazon Pet Supplies Review file retrieved from \n#https://nijianmo.github.io/amazon/index.html small subsets section\namazon_df = pd.read_json('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/AMAZON_pet_supplies/Pet_Supplies_5.json',\n                        lines=True)")


# # Reduce the data set to the most useful information for natural language processing

# In[6]:


# Remove the non-verified purchases from the data set 
display(amazon_df['verified'].value_counts())
amazon_df = amazon_df[amazon_df['verified'] == True]
amazon_df.shape


# In[7]:


# reduce the data set to the columns needed for the analysis and future identification of specific reviews
amazon_reduced = amazon_df[['reviewerID', 'overall', 'reviewText']]
display(amazon_reduced.shape)
display(amazon_reduced.isna().sum())
amazon_reduced.head()


# # Clean the data set:
# ## Ensure conformity with pythonic conventions
# ## Keep data that is relevent to the analysis at hand

# In[8]:


# remove rows where there is no review text to analyze
amazon_df = amazon_df.dropna(subset='reviewText')
display(amazon_df.isna().sum())


# In[9]:


# rename variables to conform to pythonic naming conventions and inspect
amazon_reduced.columns = ['reviewer_id', 'overall', 'review_text']


# In[10]:


# look for duplicates in the remaining data and drop them from the set
amazon_reduced = amazon_reduced.drop_duplicates()


# In[11]:


#remove records where review text is empty
amazon_reduced = amazon_reduced.dropna(subset='review_text')


# # Feature Engineering:
# ## Create features from the data set that will add to the analysis

# In[12]:


# Create a labels column where 1 is positive, 0 is neutral and -1 is negative based on overall column
amazon_reduced['labels'] = amazon_reduced.loc[:,'overall'].apply(lambda x: 1 if x > 3 else 0 if x==3 else -1)


# In[13]:


amazon_reduced['num_words'] = amazon_reduced['review_text'].apply(lambda x: len(x.split()))


# In[14]:


#Identify any characters used in each review and capture them in a separate column before removal
amazon_reduced['characters'] = amazon_reduced['review_text'].apply(lambda x: re.findall(r'[^\w\s,]',x))


# In[15]:


amazon_reduced['num_characters'] = amazon_reduced['characters'].apply(lambda x: len(x))


# In[16]:


def num_caps(review_text):
    x = sum(map(str.isupper,review_text.split()))
    return x
amazon_reduced['num_all_cap_words'] = amazon_reduced['review_text'].apply(lambda x: num_caps(x))


# # Explore the Data set:

# In[17]:


#visualize the counts and proportion of each class in labels
display(amazon_reduced['labels'].value_counts())
display(amazon_reduced['labels'].value_counts(normalize=True))
amazon_reduced['labels'].value_counts().plot.barh()
plt.title('Reviews Label Class Frequency')
plt.xlabel('Labels')
plt.ylabel('Frequency of Occurance')
plt.show()
plt.clf()


# In[18]:


#how long is the longest review included in the data set 
amazon_reduced['review_length'] = amazon_reduced['review_text'].str.len()
print('The maximum review length is {} characters long'.format(max(amazon_reduced['review_length'])))
print('The minimum review length is {} characters long'.format(min(amazon_reduced['review_length'])))


# In[19]:


#Univariate distribution of num_words (number of words in the review)
plt.hist(amazon_reduced['num_words'], bins=300)
plt.title('Number of Words Per-Review')
plt.xlabel('Count')
plt.ylabel('Frequency of Count')
plt.show()
plt.clf()
amazon_reduced['num_words'].describe()


# In[20]:


#Univariate distribution of num_characters (number of characters in the review)
plt.hist(amazon_reduced['num_characters'], bins=300)
plt.title('Number of Characters Per-Review')
plt.xlabel('Count')
plt.ylabel('Frequency of Count')
plt.show()
plt.clf()
amazon_reduced['num_characters'].describe()


# In[21]:


#Univariate distribution of num_all_cap_words (number of words in the review)
plt.hist(amazon_reduced['num_all_cap_words'], bins=300)
plt.title('Number of Capitalized Words Per-Review')
plt.xlabel('Count')
plt.ylabel('Frequency of Count')
plt.show()
plt.clf()
amazon_reduced['num_all_cap_words'].describe()


# In[22]:


#length of each review in characters
#Univariate distribution of review_length (length of each review in characters)
plt.hist(amazon_reduced['review_length'], bins=300)
plt.title('Length Per-Review')
plt.xlabel('Count')
plt.ylabel('Frequency of Count')
plt.show()
plt.clf()

amazon_reduced['review_length'].describe()


# In[23]:


#function to find unique characters in a column of text
def unique_char(text_col):
    char_list = []
    for i in text_col:
        x = re.findall(r'[^\w\s,]',i)
        for char in x:
            if char not in char_list:
                char_list.append(char)
    return char_list
                
unique_characters = unique_char(amazon_reduced['review_text']) 
print(unique_characters)


# In[24]:


amazon_reduced.head(1)


# # Balance the data set for use in training a Neural Network

# In[25]:


#produce a new dataframe balanced for each label for use in NLP
df1 = amazon_reduced.groupby('labels').apply(lambda x: x.sample(n=100000, random_state=seed)).reset_index(drop=True)


# In[26]:


df1.head(1)


# In[ ]:





# In[27]:


#Inspect the characters that will need to be removed moving forward
new_unique_characters = unique_char(df1['review_text'])
print(new_unique_characters)


# # Transform the Review column into cleaned tokens

# In[28]:


#Function to clean and tokenize the review text to improve vectorization results
def clean_token(txt_col):
    stemmer = SnowballStemmer(language='english')
    """Takes a pandas dataframe column of text data as txt_col and returns it with html removed,
    all lower case, without punctuation, numbers removed, word_tokenized, english stopwords removed,
    stemmed using Snowball stemmer. Must import SnowballStemmer for the function to work"""
    
    #remove html from text reviews
    clean_col = txt_col.apply(lambda x: BeautifulSoup(x).get_text())
    
    #convert all of the reviews to lower case
    clean_col = clean_col.str.lower()
    
    #remove punctuation from the review text column set regex to true to avoid future warning
    clean_col = clean_col.str.replace(r'[^\w\s]+','', regex=True)
    
    #remove any digits that may be included in a text review
    clean_col = clean_col.str.replace(r'\d+', '', regex=True)
    
    #tokenize words in reviews
    clean_col = clean_col.apply(lambda x: word_tokenize(x))
    
    #remove stopwords
    clean_col = clean_col.apply(lambda x:[word for word in x if word not in set(stopwords.words('english'))])
    
    #stem every word in every list of words 
    clean_col = clean_col.apply(lambda x: [stemmer.stem(w) for w in x])
    
    return clean_col


# In[29]:


get_ipython().run_cell_magic('time', '', "%%notify\ndf1['tokens'] = clean_token(df1['review_text'])")


# In[30]:


#create a column for the number of tokens in the cleaned review
df1['num_tokens'] = df1['tokens'].apply(lambda x: len(x))
plt.hist(df1['num_tokens'], bins=50)
plt.show()
plt.clf()
df1['num_tokens'].describe()


# # Max Embedding Length:
# ## Mean number of tokens: 
# ## Standard Deviation of Tokens:
# ## Max Length = Mean + (3)Standard Deviation
# ### The remainder are outliers and are likely non informative
# 

# In[31]:


get_ipython().run_cell_magic('time', '', "%%notify\n#encode the labels column into the three output clases for the neural network\ndf1 = pd.get_dummies(df1, columns=['labels'], prefix=['label'])")


# In[32]:


df1.head(1)


# # Text Representation:
# ## Use the pre-trained Google News Vectors 
# ### Convert tokens to vectors

# In[33]:


get_ipython().run_cell_magic('time', '', "%%notify\n#Load W2V model \nw2v_model = KeyedVectors.load_word2vec_format('/Users/benjaminmcdaniel/Desktop/dl_weights/GoogleNews-vectors-negative300.bin', binary=True)")


# In[34]:


#Examine the length of the vocabulary produced
print('The size of the vocabulary after training is: {} tokens'.format(len(w2v_model)))


# In[35]:


#visualize the keyed index of the vocabulary generated
print(w2v_model.get_vecattr('cat', 'count'))


# In[36]:


df1.head(1)


# # Assign the tokens to the features variable X
# # Assign the encoded labels to the target variable y

# In[37]:


X = df1['tokens']
y = y = df1[['label_-1', 'label_0', 'label_1']]


# # A function to split the variables into Training, Validation, & Test sets

# In[38]:


def train_val_test_split(X,y,train_size,random_state):
    """ Takes the feature array, target array, training set size, random_state. 
    OUTPUT FORMAT:X_train, X_val, X_test, y_train, y_val, y_test, shape 
    This function requires sklearn.model_selection.train_test_split, and conducts two train_test_splits
    that are shuffled. The output is a training set of selected size as X_train and y_train, validation and test sets
    that are made up of a 50/50 split of the remaining data as X_val, y_val, X_test, y_test respectively. The final
    output is the shape of the resulting split sets as shape. """
    
    #split one
    X_train, X_valtest, y_train, y_valtest = train_test_split(X,y,train_size=train_size,random_state=random_state, shuffle=True)
    
    #split two
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size = 0.5,random_state=random_state, shuffle=True)
    
    #print out the shape of the  resulting splits for comparison
    shape = print('The shape of X_train: {}'.format(X_train.shape)),print('The shape of y_train: {}'.format(y_train.shape)),print('The shape of X_val: {}'.format(X_val.shape)),print('The shape of y_val: {}'.format(y_val.shape)),print('The shape of X_test: {}'.format(X_test.shape)),print('The shape of y_test: {}'.format(y_test.shape))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, shape


# In[39]:


#Split the data into 80/10/10 Train, Validate, Test
X_train, X_val, X_test, y_train, y_val, y_test, shape = train_val_test_split(X,y,train_size=.8,random_state=88)


# In[ ]:





# # Save the split data for future use and submission

# In[40]:


X_train.to_csv('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/X_train.csv', index=False)
y_train.to_csv('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/y_train.csv', index=False)
X_val.to_csv('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/X_val.csv', index=False)
y_val.to_csv('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/y_val.csv', index=False)
X_test.to_csv('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/X_test.csv', index=False)
y_test.to_csv('/Users/benjaminmcdaniel/Desktop/D213/d213_task2/y_test.csv', index=False)


# # Transform the Feature Data:
# ## A function to vactorize the max sequence length of tokens:
# ### Post vector padding is utilized
# ### Vector size is 300
# ### length of the embedding is max_len: 113

# In[41]:


#function to vectorize the tokens and pad smaller sequences with zero's
def make_vectors(token_col, vect_size, max_len, model):
    """INPUT: token_col(pandas column that contains tokenized items), vect_size(size of the vector required)
    , max_len(maximum length of the embedding), model(trained word2vec embedding model used)
    OUTPUT: an array of padded vectors for use in ML"""
    vectors = []
    padding = [0.0] * vect_size
    for x in token_col:
        token_vectors = []
        count = 0
        for token in x:
            if count >= max_len:
                break
            if token in model:
                token_vectors.append(model[token])
            count += 1
        if len(token_vectors) < max_len:
            fill = max_len - len(token_vectors)
            for _ in range(fill):
                token_vectors.append(padding)
        vectors.append(token_vectors)
    
    return vectors


# In[42]:


get_ipython().run_cell_magic('time', '', "%%notify\n#vectorize training set\nX_train = np.array(make_vectors(X_train, vect_size=300, max_len = 113 , model=w2v_model)).astype('float32')")


# In[43]:


get_ipython().run_cell_magic('time', '', "%%notify\n#vectorize validation set\nX_val = np.array(make_vectors(X_val, vect_size=300, max_len = 113 , model=w2v_model)).astype('float32')")


# In[44]:


get_ipython().run_cell_magic('time', '', "%%notify\n#vectorize test set\nX_test = np.array(make_vectors(X_test, vect_size=300, max_len = 113 , model=w2v_model)).astype('float32')")


# # Visualize padded vector sequence:

# In[46]:


print(X_train[4])


# # Build a CNN LSTM model for multiclass text sentiment classification

# In[47]:


filters = 5
kernel_sz = 5
hid_lay_1nodes = 200
hid_lay_2nodes = 200
hid_lay_3nodes = 200
hid_lay_4nodes = 200
hid_lay_5nodes = 100
hid_lay_6nodes = 100
dropout=0.2
num_epoch=100
batch_sz=32


# In[48]:


#Set early stopping parameters for validation set accuracy, max, and patience of 3
stop = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose=1, patience=3)


# In[49]:


#create a file to save best model weights for future use
hf = h5py.File('/Users/benjaminmcdaniel/Desktop/dl_weights/cnn_weights_best.h5', 'w')
hf.close()


# In[50]:


#Assemble the model layers 
cnn_model = Sequential()
cnn_model.add(Conv1D(32, 5,strides=1, padding='same', activation = 'relu', input_shape = (113,300)))
cnn_model.add(MaxPooling1D(5))

cnn_model.add(Dense(hid_lay_1nodes, activation ='relu'))
cnn_model.add(Dropout(dropout))

cnn_model.add(Dense(hid_lay_2nodes, activation = 'relu'))
cnn_model.add(Dropout(dropout))

cnn_model.add(Conv1D(100, kernel_sz, activation='relu'))
cnn_model.add(Dropout(dropout))
cnn_model.add(MaxPooling1D(5))

cnn_model.add(Dense(hid_lay_3nodes, activation = 'relu'))
cnn_model.add(Dropout(dropout))


cnn_model.add(Dense(hid_lay_4nodes, activation = 'relu'))
cnn_model.add(Dropout(dropout))

cnn_model.add(Dense(hid_lay_5nodes, activation = 'relu'))
cnn_model.add(Dropout(dropout))

cnn_model.add(LSTM(128, activation='relu', return_sequences=True))


cnn_model.add(Dense(hid_lay_6nodes, activation ='relu'))
cnn_model.add(GlobalMaxPooling1D())

cnn_model.add(Dense(3, activation = 'softmax'))
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(cnn_model.summary())
filepath="/Users/benjaminmcdaniel/Desktop/dl_weights/cnn_weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
stop = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose=1, patience=3)
callbacks_list = [checkpoint, stop]
history = cnn_model.fit(X_train, y_train, epochs=num_epoch, batch_size=batch_sz,verbose = 1,callbacks = callbacks_list,validation_data=(X_val,y_val))


# # Visualize Model Performance 

# In[51]:


#visualize training accuracy vs validation accuracy
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Traning vs Validation Accuracy')
plt.legend()
plt.show()
plt.clf()


# In[52]:


#visualize training loss vs validation loss
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Traning vs Validation Loss')
plt.legend()
plt.show()
plt.clf()


# In[53]:


#Evaluate the model on the test set
cnn_model.load_weights('/Users/benjaminmcdaniel/Desktop/dl_weights/cnn_weights_best.h5')
scores = cnn_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# # Save the Model:

# In[54]:


# Expressly save the model
hf = h5py.File('/Users/benjaminmcdaniel/Desktop/dl_weights/cnn_lstm.h5', 'w')
hf.close()
from keras.models import load_model
cnn_model.save('/Users/benjaminmcdaniel/Desktop/dl_weights/cnn_lstm.h5')


# In[ ]:




