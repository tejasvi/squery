import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from itertools import chain
from collections import Counter
import regex as re

def load():
    t = pd.read_pickle("train.pkl")
    with open('freq.pkl', 'rb') as f:
        frequencies = pickle.load(f)
    with open('dlist.pkl', 'rb') as f:
        dlist = pickle.load(f)
    model = Word2Vec.load("word2vec.model")
    return t, model, frequencies, dlist

def preproc(s):
    return [
        x
        for x in simple_preprocess(s)
        if x
        not in {
            "gm",
            "ml",
            "kg",
            "with",
            "for",
            "the",
            "rs",
            "of",
            "under",
            "less",
            "more",
            "than",
            "lower",
            "greater",
        }
    ]

def autocomplete(q, freq):
    return next((k for k in freq if k.startswith(q)), None)

def run_sif(query, sentences2, model, freqs={}, a=0.001):
    total_freq = sum(freqs.values())
    embeddings = []

    tokens1 = [token if token in model.wv else autocomplete(token, freqs) for token in query]
    if not tokens1:
        return None
    for i in range(tokens1.count(None)): tokens1.remove(None)
    weights1 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens1]
    embedding1 = np.zeros((len(sentences2), model.trainables.layer1_size)) + np.average(
        [model.wv[token] for token in tokens1], axis=0, weights=weights1
    )

    embedding2 = np.zeros((len(sentences2), model.trainables.layer1_size))

    # SIF requires us to first collect all sentence embeddings and then perform
    # common component analysis.
    for i, sent2 in enumerate(sentences2):

        tokens2 = [token for token in sent2 if token in model.wv]
        n = len(set(tokens1) & set(tokens2)) / len(tokens1)

        weights2 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens2]
        embedding2[i] = np.average(
            [model.wv[token] for token in tokens2], axis=0, weights=weights2
        )

        embedding1[i] += 15 * n * embedding2[i]

    sims = np.einsum("ij,ij->i", embedding1, embedding2) / (
        np.linalg.norm(embedding1, axis=1) * np.linalg.norm(embedding2, axis=1)
    )

    return sims


def wt(q):
    w = i = None
    for i, x in enumerate(q):
        if x in ("gm", "ml", "kg", "g", "l", "lt", "ltr", "ml", "pcs", "xgm",):
            if i:
                try:
                    w = int(float(q[i - 1]))
                except:
                    pass
    return w


def cost(q):
    more = c = i = None
    for i, x in enumerate(q):
        if x in ("less", "lower"):
            if i < len(q) - 2:
                try:
                    c = int(float(q[i + 2]))
                    more = 1
                except:
                    pass
        elif x == "under":
            if i < len(q) - 1:
                try:
                    c = int(float(q[i + 1]))
                    more = 1
                except:
                    pass
        elif x in ("more", "greater"):
            if i < len(q) - 2:
                try:
                    c = int(float(q[i + 2]))
                    more = 3
                except:
                    pass
        if x == "rs":
            if i:
                try:
                    c = int(float(q[i - 1]))
                    more = 2
                except:
                    pass
            else:
                try:
                    c = int(float(q[i + 1]))
                    more = 2
                except:
                    pass
        if c:
            break
    return more, c


def run(q, boost=[], b=1, n=10):
    qcheck = re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", q.lower()).strip().split()
    grammage = wt(qcheck)
    op, price = cost(qcheck)

    q += 4 * int(b) * (" " + " ".join(boost))
    scores = run_sif(preproc(q), dlist, freqs=frequencies, model=model)
    df = t.copy()
    df["scores"] = scores

    # price
    if price:
        if op == 1:
            df.loc[df["Final Price"] < price, "scores"] += 0.005
        elif op == 2:
            df.loc[df["Final Price"].between(price - 10, price + 10), "scores"] += 0.005
        elif op == 3:
            df.loc[df["Final Price"] > price, "scores"] += 0.005

    # grammage
    if grammage:
        df.loc[df["wt"] == grammage, "scores"] += 0.005

    return df


if __name__ == "__main__":
    t, model, frequencies, dlist = load()


    '''
    # SQuery
    _-team DataMinds_
    '''
    query = st.text_input('Enter query and press enter', 'Powder 250 gm under 150 rs')
    n = st.number_input('Number of results', 1, 999, 10)
    boost = st.text_input('Enter space separated boosted brands (optional)', 'ayghd')
    boost = boost.split()
    b = st.number_input('Boost extent', 0, 5, 1)
    df = run(query, boost=boost, b=b)
    temp=df.sort_values("scores", ascending=False)[
        ["Product Description", "Grammage", "Final Price"]
    ].head(n)
    temp.index = np.arange(1, len(temp) + 1)
    st.table(temp)
    '''
    ## Sample data
    '''
    st.button('Refresh')
    sample = t.sample(n=10)[
        ["Product Description", "Grammage", "Final Price"]
    ].head(n)
    sample.index = np.arange(1, len(sample) + 1)
    st.table(sample)



