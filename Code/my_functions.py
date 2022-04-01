# The following packages, modules and functions are used for processing the data

# OS for path
import os
import requests
import gzip
import io

# General packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
import csv


from sklearn.model_selection import train_test_split, GridSearchCV

# NLTK
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Gensim: Preprocessing and Modelling
from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, stem_text
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# Metrics
from scipy.spatial.distance import cosine
