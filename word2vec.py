import tensorflow
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model

def word_vec():
    model=keras.models.load_model("models\Word2vec.h5")
    return model