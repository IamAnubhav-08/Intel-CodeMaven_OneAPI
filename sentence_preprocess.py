# # !pip install modin[all]

# # pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -fhttps://download.pytorch.org/whl/torch_stable.html
# # pip install intel_extension_for_pytorch==1.13.100 -fhttps://developer.intel.com/ipex-whl-stable-cpu

import numpy as np
import pandas
import argparse
import sys
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
# # from word2vec import word_vec
import tensorflow
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model


# def text_preprocess(sent_type="",Sent=""):
#   model=keras.models.load_model("models/Word2vec.h5")
#   # model._make_predict_function()
#   # info = api.info()  # show info about available models/datasets
#   nltk.download('punkt')
#   title_words = nltk.word_tokenize(Sent.lower())
#   headline_words = nltk.word_tokenize(Sent.lower())

#   nltk.download('stopwords')
#   stop_words = set(stopwords.words('english'))
  
#   Sub_List_title=[]
#   for word in headline_words:
#     if word not in stop_words and word.isalnum():
#       Sub_List_title.append(word)


#   word_vectors_mat_title=[]
#   if sent_type=="Title":
#     if len(Sub_List_title)>18 :
#       word_vectors_mat_title = [model[Sub_List_title[i]] for i in range(18) if Sub_List_title[i] in model.key_to_index]
#     else:
#       word_vectors_mat_title = [model[words] for words in Sub_List_title if words in model.key_to_index]

#     sentence_embedding = [np.mean(word_vectors_mat_title) for words in word_vectors_mat_title]

#     sentence_embedding_mat_title=np.array(sentence_embedding)

#   elif sent_type=="Headline":
#     if len(Sub_List_title)>48 :
#       word_vectors_mat_title = [model[Sub_List_title[i]] for i in range(48) if Sub_List_title[i] in model.key_to_index]
#     else:
#       word_vectors_mat_title = [model[words] for words in Sub_List_title if words in model.key_to_index]

#     sentence_embedding = [np.mean(word_vectors_mat_title) for words in word_vectors_mat_title]

#     sentence_embedding_mat_title=np.array(sentence_embedding)

#   sentence_embedding = list(sentence_embedding_mat_title)
#   sent_size = len(sentence_embedding)
#   if sent_type=="Title":
#     max_length=18
#   else:
#     max_length=48
#   if max_length-sent_size > 0:
#    sentence_embedding = sentence_embedding + [0]*(max_length - sent_size) 
  
#   final_sent_mat_title = np.array(sentence_embedding)
#   return final_sent_mat_title

import numpy as np
import pandas
import argparse
import sys
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords

def text_preprocess(sent_type="",Sent=""):


  info = api.info()  # show info about available models/datasets
  model = api.load("word2vec-google-news-300") 
  nltk.download('punkt')
  
  title_words = nltk.word_tokenize(Sent.lower())
  headline_words = nltk.word_tokenize(Sent.lower())

  nltk.download('stopwords')
  stop_words = set(stopwords.words('english'))
  
  Sub_List_title=[]
  for word in headline_words:
    if word not in stop_words and word.isalnum():
      Sub_List_title.append(word)

  word_vectors_mat_title=[]
  if sent_type=="Title":
    if len(Sub_List_title)>18 :
      word_vectors_mat_title = [model[Sub_List_title[i]] for i in range(18) if Sub_List_title[i] in model.key_to_index]
    else:
      word_vectors_mat_title = [model[words] for words in Sub_List_title if words in model.key_to_index]

    sentence_embedding = [np.mean(word_vectors_mat_title) for words in word_vectors_mat_title]

    sentence_embedding_mat_title=np.array(sentence_embedding)

  elif sent_type=="Headline":
    if len(Sub_List_title)>48 :
      word_vectors_mat_title = [model[Sub_List_title[i]] for i in range(48) if Sub_List_title[i] in model.key_to_index]
    else:
      word_vectors_mat_title = [model[words] for words in Sub_List_title if words in model.key_to_index]

    sentence_embedding = [np.mean(word_vectors_mat_title) for words in word_vectors_mat_title]

    sentence_embedding_mat_title=np.array(sentence_embedding)

  sentence_embedding = list(sentence_embedding_mat_title)
  sent_size = len(sentence_embedding)
  if sent_type=="Title":
    max_length=18
  else:
    max_length=48
  if max_length-sent_size > 0:
   sentence_embedding = sentence_embedding + [0]*(max_length - sent_size) 
  
  final_sent_mat_title = np.array(sentence_embedding)
  return final_sent_mat_title






















































