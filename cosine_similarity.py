import os
import re
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim

def cleaning(text):
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
        if lemmatized_token not in gensim.parsing.preprocessing.STOPWORDS:
            processed_tokens.append(lemmatized_token)
    return processed_tokens

def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        # Add your text extraction logic here
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
    corpus = []  # Initialize an empty list to store the extracted text

    for root, dirs, files in os.walk(folder_path):
        folder_documents = []  # Initialize a list to store documents within the current folder

        for file in files:
            file_path = os.path.join(root, file)
            extracted_text = extract_text_from_file(file_path)
            if extracted_text:
                folder_documents.append(extracted_text)

        if folder_documents:
            corpus.append(folder_documents)  # Add the list of documents within the current folder to the corpus

    return corpus

def calculate_cosine_similarity(document_set1, document_set2):
    vectorizer = TfidfVectorizer()
    preprocessed_documents_set1 = []
    preprocessed_documents_set2 = []
    for folder in document_set1:
        for doc in folder:
            if not isinstance(doc, str):
                print(f"Warning: Document is not a string: {doc}")
            else:
                cleaned_doc = cleaning(doc)
                preprocessed_doc = preprocessing(cleaned_doc)
                preprocessed_text = ' '.join(preprocessed_doc)
                preprocessed_documents_set1.append(preprocessed_text)
    for folder in document_set2:
        for doc in folder:
            if not isinstance(doc, str):
                print(f"Warning: Document is not a string: {doc}")
            else:
                cleaned_doc = cleaning(doc)
                preprocessed_doc = preprocessing(cleaned_doc)
                preprocessed_text = ' '.join(preprocessed_doc)
                preprocessed_documents_set2.append(preprocessed_text)
    tfidf_matrix_set1 = vectorizer.fit_transform(preprocessed_documents_set1)
    print(tfidf_matrix_set1)
    tfidf_matrix_set2 = vectorizer.transform(preprocessed_documents_set2)

    cosine_similarity_matrix1 = cosine_similarity(tfidf_matrix_set1)
    cosine_similarity_matrix2 = cosine_similarity(tfidf_matrix_set2)
    cosine_similarity_matrix_between_sets = cosine_similarity(tfidf_matrix_set1, tfidf_matrix_set2)
    np.set_printoptions(threshold=np.inf)

    print("Cosine Similarity Matrix within document_set1:\n", cosine_similarity_matrix1)
    print("Cosine Similarity Matrix within document_set2:\n", cosine_similarity_matrix2)
    print("Cosine Similarity Matrix between document_set1 and document_set2:\n", cosine_similarity_matrix_between_sets)

    # Mean of all the cosine similarity values
    average_similarity_score_set1 = cosine_similarity_matrix1.mean()
    average_similarity_score_set2 = cosine_similarity_matrix2.mean()
    average_similarity_score_between_sets = cosine_similarity_matrix_between_sets.mean()

    print("Average Cosine Similarity Matrix within document_set1:", average_similarity_score_set1)
    print("Average Cosine Similarity Matrix within document_set2:", average_similarity_score_set2)
    print("Average Cosine Similarity Matrix between document_set1 and document_set2:", average_similarity_score_between_sets)

    return cosine_similarity_matrix1, cosine_similarity_matrix2, cosine_similarity_matrix_between_sets

folder_path1 = 'movements/American Gothic/' # choose the ones needed
folder_path2 = 'movements/Trascendentalism/'
corpus1 = build_corpus(folder_path1)
corpus2 = build_corpus(folder_path2)

calculate_cosine_similarity(corpus1, corpus2)






