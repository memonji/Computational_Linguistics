import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def cleaning(text):
    # text = np.char.lower(text)
    text = text.lower()
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for symbol in symbols:
        text = text.replace(symbol, ' ')
    text = text.replace("'", "")
    words = text.split()
    new_text = ""
    for word in words:
        if len(word) > 1:
            new_text = new_text + " " + word
    return new_text
def preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    processed_tokens = []
    for token in gensim.utils.simple_preprocess(text):
        lemmatized_token = lemmatizer.lemmatize(token, pos='v')
        processed_tokens.append(lemmatized_token)
        # if lemmatized_token not in gensim.parsing.preprocessing.STOPWORDS:
        #     processed_tokens.append(lemmatized_token)
    return processed_tokens

def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        starting_sentence = r"\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK"
        starting_sentence = re.compile(starting_sentence, re.IGNORECASE)
        ending_sentence = r"\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG"
        ending_sentence = re.compile(ending_sentence)
        starting_sentence_match = re.search(starting_sentence, text)
        ending_sentence_match = re.search(ending_sentence, text)
        if starting_sentence_match and ending_sentence_match:
            starting_sentence_index = starting_sentence_match.end()
            ending_sentence_index = ending_sentence_match.start()
            extracted_text = text[starting_sentence_index:ending_sentence_index].strip()
            extracted_text = extracted_text.replace('\n', ' ')
            return extracted_text
    return None
def build_corpus(folder_path):
    corpus = []
    for root, dirs, files in os.walk(folder_path):
        folder_documents = []
        for file in files:
            file_path = os.path.join(root, file)
            extracted_text = extract_text_from_file(file_path)
            if extracted_text:
                folder_documents.append(extracted_text)
        if folder_documents:
            corpus.append(folder_documents)
    return corpus

def tfidf_corpus_nouns(documents, num_top_words, flag=False):
    # stop_words = list(ENGLISH_STOP_WORDS)
    # vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectorizer = TfidfVectorizer()
    if flag:
        print('preprocessed')
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                if not isinstance(doc, str):
                    print(f"Warning: Document is not a string: {doc}")
                else:
                    cleaned_doc = cleaning(doc)
                    preprocessed_doc = preprocessing(cleaned_doc)
                    preprocessed_text = ' '.join(preprocessed_doc)
                    preprocessed_documents.append(preprocessed_text)
    else:
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                preprocessed_text = ' '.join(doc)
                preprocessed_documents.append(preprocessed_text)
    if flag:
        preprocessed_documents_nouns = [' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "NOUN" and not token.is_stop]) for doc in
                                    preprocessed_documents]
    else:
        preprocessed_documents_nouns = [
            ' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "NOUN"]) for doc in
            preprocessed_documents]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents_nouns)
    feature_names = vectorizer.get_feature_names_out()
    feature_indices = {feature: idx for idx, feature in enumerate(feature_names)}
    nouns_feature_names = set()
    for doc in preprocessed_documents_nouns:
        nouns_feature_names.update([feature for feature in doc.split() if feature in feature_indices])
    top_nouns = sorted(nouns_feature_names, key=lambda x: np.sum(tfidf_matrix[:, feature_indices[x]].toarray()),
                       reverse=True)[:num_top_words]
    top_noun_tfidf_values = [(noun, tfidf_matrix[:, feature_indices[noun]].toarray().sum()) for noun in top_nouns]
    print("Top TF-IDF Nouns:")
    for noun, tfidf_value in top_noun_tfidf_values:
        print(f"Feature: {noun} (Noun), TF-IDF Value: {tfidf_value:.4f}")
def tfidf_corpus_adverbs(documents, num_top_words, flag=False):
    # stop_words = list(ENGLISH_STOP_WORDS)
    # vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectorizer = TfidfVectorizer()
    if flag:
        print('preprocessed')
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                if not isinstance(doc, str):
                    print(f"Warning: Document is not a string: {doc}")
                else:
                    cleaned_doc = cleaning(doc)
                    preprocessed_doc = preprocessing(cleaned_doc)
                    preprocessed_text = ' '.join(preprocessed_doc)
                    preprocessed_documents.append(preprocessed_text)
    else:
        preprocessed_documents = documents

    if flag:
        preprocessed_documents_adverbs = [' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "ADV" and not token.is_stop]) for doc in
                                    preprocessed_documents]
    else:
        preprocessed_documents_adverbs = [
            ' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "ADV"]) for doc in
            preprocessed_documents]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents_adverbs)
    feature_names = vectorizer.get_feature_names_out()
    feature_indices = {feature: idx for idx, feature in enumerate(feature_names)}
    adverbs_feature_names = set()
    for doc in preprocessed_documents_adverbs:
        adverbs_feature_names.update([feature for feature in doc.split() if feature in feature_indices])
    top_adverbs = sorted(adverbs_feature_names, key=lambda x: np.sum(tfidf_matrix[:, feature_indices[x]].toarray()),
                         reverse=True)[:num_top_words]
    top_adverb_tfidf_values = [(adverb, tfidf_matrix[:, feature_indices[adverb]].toarray().sum()) for adverb in
                               top_adverbs]
    print("Top TF-IDF Adverbs:")
    for adverb, tfidf_value in top_adverb_tfidf_values:
        print(f"Feature: {adverb} (Adverb), TF-IDF Value: {tfidf_value:.4f}")
def tfidf_corpus_verbs(documents, num_top_words, flag=False):
    # stop_words = list(ENGLISH_STOP_WORDS)
    # vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectorizer = TfidfVectorizer()
    if flag:
        print('preprocessed')
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                if not isinstance(doc, str):
                    print(f"Warning: Document is not a string: {doc}")
                else:
                    cleaned_doc = cleaning(doc)
                    preprocessed_doc = preprocessing(cleaned_doc)
                    preprocessed_text = ' '.join(preprocessed_doc)
                    preprocessed_documents.append(preprocessed_text)
    else:
        preprocessed_documents = documents
    if flag:
        preprocessed_documents_verbs = [' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "VERB" and not token.is_stop]) for doc in
                                    preprocessed_documents]
    else:
        preprocessed_documents_verbs = [
            ' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "VERB"]) for doc in
            preprocessed_documents]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents_verbs)
    feature_names = vectorizer.get_feature_names_out()
    feature_indices = {feature: idx for idx, feature in enumerate(feature_names)}
    verbs_feature_names = set()
    for doc in preprocessed_documents_verbs:
        verbs_feature_names.update([feature for feature in doc.split() if feature in feature_indices])
    top_verbs = sorted(verbs_feature_names, key=lambda x: np.sum(tfidf_matrix[:, feature_indices[x]].toarray()),
                       reverse=True)[:num_top_words]
    top_verb_tfidf_values = [(verb, tfidf_matrix[:, feature_indices[verb]].toarray().sum()) for verb in top_verbs]
    print("Top TF-IDF Verbs:")
    for verb, tfidf_value in top_verb_tfidf_values:
        print(f"Feature: {verb} (Verb), TF-IDF Value: {tfidf_value:.4f}")
def tfidf_corpus_adj(documents, num_top_words, flag=False):
    # stop_words = list(ENGLISH_STOP_WORDS)
    # vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectorizer = TfidfVectorizer()
    if flag:
        print('preprocessed')
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                if not isinstance(doc, str):
                    print(f"Warning: Document is not a string: {doc}")
                else:
                    cleaned_doc = cleaning(doc)
                    preprocessed_doc = preprocessing(cleaned_doc)
                    preprocessed_text = ' '.join(preprocessed_doc)
                    preprocessed_documents.append(preprocessed_text)
    else:
        preprocessed_documents = documents
    if flag:
        preprocessed_documents_adj = [
            ' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "ADJ" and not token.is_stop]) for doc in
            preprocessed_documents]
    else:
        preprocessed_documents_adj = [
            ' '.join([token.lemma_ for token in nlp(doc) if token.pos_ == "ADJ"]) for doc in
            preprocessed_documents]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents_adj)
    feature_names = vectorizer.get_feature_names_out()
    feature_indices = {feature: idx for idx, feature in enumerate(feature_names)}
    adj_feature_names = set()
    for doc in preprocessed_documents_adj:
        adj_feature_names.update([feature for feature in doc.split() if feature in feature_indices])
    top_adjectives = sorted(adj_feature_names, key=lambda x: np.sum(tfidf_matrix[:, feature_indices[x]].toarray()),
                            reverse=True)[:num_top_words]
    top_adj_tfidf_values = [(adj, tfidf_matrix[:, feature_indices[adj]].toarray().sum()) for adj in top_adjectives]
    print("Top TF-IDF Adjectives:")
    for adj, tfidf_value in top_adj_tfidf_values:
        print(f"Feature: {adj} (Adjective), TF-IDF Value: {tfidf_value:.4f}")
def tfidf_multiple_counter(documents, num_top_words, flag=False, verbose=False):
    stop_words = list(ENGLISH_STOP_WORDS)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    # vectorizer = TfidfVectorizer()
    if flag:
        print('preprocessed')
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                if not isinstance(doc, str):
                    print(f"Warning: Document is not a string: {doc}")
                else:
                    cleaned_doc = cleaning(doc)
                    preprocessed_doc = preprocessing(cleaned_doc)
                    preprocessed_text = ' '.join(preprocessed_doc)
                    preprocessed_documents.append(preprocessed_text)
    else:
        preprocessed_documents = []
        for folder in documents:
            for doc in folder:
                # preprocessed_text = ' '.join(doc)
                preprocessed_documents.append(doc)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
    feature_names = vectorizer.get_feature_names_out()
    total_word_tfidf = tfidf_matrix.sum(axis=0)
    total_word_tfidf = total_word_tfidf.A1
    top_word_indices = total_word_tfidf.argsort()[::-1][:num_top_words]
    top_words = [feature_names[i] for i in top_word_indices]
    top_tfidf_scores = [total_word_tfidf[i] for i in top_word_indices]
    if verbose:
        print("Top Words by TF-IDF Score for the Entire Corpus:")
        for word, tfidf_score in zip(top_words, top_tfidf_scores):
            print(f'{word}: {tfidf_score:.4f}')
    return top_words, top_tfidf_scores

num_top_words = 100
folder_path = 'movements/American Gothic/' # whatever
corpus = build_corpus(folder_path)

tfidf_corpus_nouns(corpus, num_top_words, flag=True)
tfidf_corpus_adverbs(corpus, num_top_words, flag=True)
tfidf_corpus_verbs(corpus, num_top_words, flag=True)
tfidf_corpus_adj(corpus, num_top_words, flag=True)
# tfidf_multiple_counter(corpus, num_top_words = 100, flag=True, verbose = True)
