import os
import re

from collections import Counter
import syllables
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 90223111
nltk.download('wordnet')
import gensim
from textstat import textstat
from textstat.textstat import legacy_round


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
    corpus = []
    doc_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_text = extract_text_from_file(file_path)
            if extracted_text:
                corpus.append(extracted_text)
                doc_paths.append(file_path)
    return corpus, doc_paths

def preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    processed_tokens = []
    for token in gensim.utils.simple_preprocess(text):
        lemmatized_token = lemmatizer.lemmatize(token, pos='v')
        if lemmatized_token not in gensim.parsing.preprocessing.STOPWORDS:
            processed_tokens.append(lemmatized_token)
    return processed_tokens

def some_averages(text):
    def break_sentences(text):
        doc = nlp(text)
        return list(doc.sents)
    def word_count(text):
        sentences = break_sentences(text)
        words = 0
        for sentence in sentences:
            words += len([token for token in sentence])
        return words
    def sentence_count(text):
        sentences = break_sentences(text)
        return len(sentences)
    def avg_sentence_length(text):
        words = word_count(text)
        sentences = sentence_count(text)
        average_sentence_length = float(words / sentences)
        return average_sentence_length
    def syllables_count(word):
        return textstat.syllable_count(word)
    def avg_syllables_per_word(text):
        syllable = syllables_count(text)
        words = word_count(text)
        ASPW = float(syllable) / float(words)
        return legacy_round(ASPW, 1)
    def difficult_words(text):
        doc = nlp(text)
        words = []
        sentences = break_sentences(text)
        for sentence in sentences:
            words += [str(token) for token in sentence]
        diff_words_set = set()
        for word in words:
            syllable_count = syllables_count(word)
            if word not in nlp.Defaults.stop_words and syllable_count >= 2:
                diff_words_set.add(word)
        return len(diff_words_set)
    def flesch_reading_ease(text):
        FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - \
              float(84.6 * avg_syllables_per_word(text))
        return legacy_round(FRE, 2)
    def gunning_fog(text):
        per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
        grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
        return grade
    def dale_chall_readability_score(text):
        words = word_count(text)
        count = word_count(text) - difficult_words(text)
        if words > 0:
            per = float(count) / float(words) * 100
        diff_words = 100 - per
        raw_score = (0.1579 * diff_words) + \
                    (0.0496 * avg_sentence_length(text))
        if diff_words > 5:
            raw_score += 3.6365
        return raw_score
    def calculate_flesch_kincaid(text):
        sentences = nltk.sent_tokenize(text)
        total_words = len(nltk.word_tokenize(text))
        total_syllables = 0
        for word in nltk.word_tokenize(text):
            total_syllables += syllables.estimate(word)
        average_words_per_sentence = total_words / len(sentences)
        flesch_kincaid_score = 0.39 * average_words_per_sentence + 11.8 * (total_syllables / total_words) - 15.59
        return flesch_kincaid_score
    def calculate_ari(text):
        sentences = text.split('.')
        total_chars = sum(len(sentence) for sentence in sentences)
        total_words = sum(len(sentence.split()) for sentence in sentences)
        ari = (4.71 * (total_chars / total_words)) + (0.5 * (total_words / len(sentences)))
        return ari
    def find_hapax(text):
        words = text.split()
        word_counts = Counter(words)
        hapax_words = [word for word, count in word_counts.items() if count == 1]
        return len(hapax_words)
    def hlr(text):
        tokens = word_tokenize(text.lower())
        freq_dist = FreqDist(tokens)
        hapax_count = sum(1 for word, freq in freq_dist.items() if freq == 1)
        total_words = len(tokens)
        hlr = hapax_count / total_words
        return hlr
    def calculate_ttr(text):
        words = text.split()
        total_tokens = len(words)
        unique_words = set(words)
        total_types = len(unique_words)
        ttr = total_types / total_tokens
        return ttr
    stats = {
        'word count': word_count(text),
        'sentence count': sentence_count(text),
        'average sentence length': avg_sentence_length(text),
        'avg_syllables_per_word count': avg_syllables_per_word(text),
        'difficult_words': difficult_words(text),
        'flesch_reading_ease': flesch_reading_ease(text),
        'gunning_fog': gunning_fog(text),
        'dale_chall_readability_score': dale_chall_readability_score(text),
        'calculate_flesch_kincaid': calculate_flesch_kincaid(text),
        'calculate ari': calculate_ari(text),
        'find hapax': find_hapax(text),
        'hlr': hlr(text),
        'ttr': calculate_ttr(text)
    }
    return stats

folder_path = 'movements/American Gothic/' # whatever
corpus = build_corpus(folder_path)
some_averages(corpus)