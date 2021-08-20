import re
import string

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity

import pickle as pkl

# load corpus and extract documents from it
with open("models/corpus.df", "rb") as f:
    corpus = pkl.load(f)
document = corpus['question']

stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()

STOP_WORDS = [stemmer.stem(stopword)
              for stopword in stopwords.words("english")]
L_STOP_WORDS = [lemmer.lemmatize(stopword)
                for stopword in stopwords.words("english")]


def clean_text1(text):
    ''' Make texts lower case, remove text in square bracket, remove punctuation'''
    text = text.lower()
    text = re.sub(r"""[\/]""", ' or ', text)
    # Removes quotation marks.
    text = text.replace('"', "")

    # Remove numeric characters.
    text = re.sub('\w*\d\w*', ' ', text)

    # Remove punctuation.
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    return text


def tokenizer(text):

    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# TFIDF


def mods():
    tfidf = TfidfVectorizer(
        stop_words=STOP_WORDS,
        preprocessor=clean_text1,
        tokenizer=tokenizer,
        min_df=2,
        #     max_df=.80
    )

    return tfidf


tfidf = mods()


def train():
    doc_term_mtx = tfidf.fit_transform(document)
    vocab = tfidf.get_feature_names()
    model = TruncatedSVD(169)
    model.fit(doc_term_mtx)
    question_topic = model.transform(doc_term_mtx)

    return model, question_topic


model, ques_topic = train()


def predict(new_q):
    embeded_querry = tfidf.transform([new_q])
    embeded_q_topic = model.transform(embeded_querry)

    res = []
    for index in range(corpus.shape[0]):
        question, embedding = corpus['answer'][index], ques_topic[index]
        cos_sim = round(cosine_similarity(
            [embedding], embeded_q_topic)[0][0], 3)
        res.append(cos_sim)
    #     print(idx)
    #     print(sim, sent)

    n = 5
    idx_array = np.array(res).argsort()[-n:][::-1]
    answer_idx = idx_array[0]
    print(f'Top {n} clossest questions:\n')
    for i in idx_array:
        print(res[i], '---', corpus['question'][i])

    print(f'\nQ: {new_q}?')
    print(f"Matched: {corpus['question'][answer_idx]}?\n")
    print(f"Ans: {corpus['answer'][answer_idx]}?")
    ans = f"Ans: {corpus['answer'][answer_idx]}?"
    return ans


train()
predict('how do i cancel my account?')
