from deco import concurrent, synchronized
from json import loads, dumps
from tqdm import tqdm
from pathlib import Path
from pathos.multiprocessing import ProcessingPool
import numpy as np
import logging
import nltk

logging.basicConfig(level=logging.INFO)
FILE_POP = Path("pop.json")
FILE_DB = Path("db.json")
FILE_DJENT = Path("djent.json")
FILE_EMOLEX = Path("EmoLex.cs")
DIR_OUTPUT = Path("output")
EMOTION_CATEGORIES = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
POOL = ProcessingPool(4)

def get_lyrics(source):
    """
    Grabs lyrics either based on the Bilboard Top 100 or got-djent.com.
    :param source: Either "pop" or "djent"
    :return: The crawled lyrics.
    """
    logging.info("Grabbing Lyrics")
    logging.warning("NOT IMPLEMENTED")
    ...


def read_emolex():
    """
    Reads the EmoLex csv file.
    :return: A dictionary with the contents.
    """
    logging.info("Reading {}".format(FILE_EMOLEX))
    emolex = {}
    import csv
    with FILE_EMOLEX.open("r") as ef:
        reader = csv.reader(ef)
        next(reader)  # skip header
        for row in tqdm(reader):
            emolex[row[0].lower().strip()] = np.array(row[4:], dtype=int)
    return emolex


def analyze_emotions(lyrics, *, emolex=None, english_only=True):
    """
    Analyzes emotions based on EmoLex. Annotates the passed lyrics.
    :param lyrics: The lyrics.
    :param emolex: The EmoLex. If empty executes read_emolex.
    :return: Annotated lyrics
    """
    logging.info("Analyzing Emotions")
    emolex = read_emolex() if not emolex else emolex
    for idx, song in tqdm(enumerate(lyrics), total=len(lyrics)):
        if "language" in song and english_only and song["language"] != "en_US":
            continue
        emotion_vector = np.zeros(8, dtype=int)
        sentences = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(song["text_raw"])]
        for sen in sentences:
            for t in sen:
                word = t.lower().strip()
                if word in emolex:
                    emotion_vector += emolex[word]
        song["emotions"] = emotion_vector.tolist()
    return lyrics


def calculate_statistics(lyrics):
    """
    Calculates statistics based on the text_raw of the lyrics.
    :return: Annotated lyrics containing information about the songs
    """
    logging.info("Calculating Statistics")
    from textstat.textstat import textstat
    for idx, song in tqdm(enumerate(lyrics), total=len(lyrics)):
        song["num_syllables"] = textstat.syllable_count(lyrics["text_raw"])
        song["num_words"] = textstat.lexicon_count(lyrics["text_raw"])
        song["num_sentences"] = textstat.sentence_count(lyrics["text_raw"])
        song["flesch_score"] = textstat.flesch_reading_ease(lyrics["text_raw"])
        song["flesch_kincaid_level"] = textstat.flesch_kincaid_grade(lyrics["text_raw"])
        song["fog_score"] = textstat.gunning_fog(lyrics["text_raw"])
        song["num_difficult_words"] = textstat.dale_chall_readability_score(lyrics["text_raw"])
    return lyrics


def detect_language(lyrics, *, threshold=0.7):
    """
    Annotates the lyrics. Currently only detects English. If English is detected with threshold "en_US" is added, else ""
    :type threshold: How many percent of the lyrics have to be detected to count as English.
    :return: 
    """
    logging.info("Detecting language")
    import enchant
    d = enchant.Dict("en_US")
    for song in tqdm(lyrics):
        checks = [d.check(w) for w in song["text_raw"]]
        ratio = checks.count(True) / len(checks)
        song["language"] = "en_US" if ratio > 0.7 else ""
    return lyrics


def extract_keywords(lyrics, *, top_keywords=3):
    """
    Annotates the lyrics with the extracted top keywords.
    :param lyrics: The lyrics
    :param top_keywords: How many keywords to include.
    :return: Lyrics with annotated keywords.
    """
    logging.info("Extracting Keywords")
    from rake_nltk import Rake
    r = Rake()
    for idx, song in tqdm(enumerate(lyrics), total=len(lyrics)):
        r.extract_keywords_from_text(song["text_raw"])
        song["keywords"] = r.get_ranked_phrases()[:top_keywords]
    return lyrics

def find_topics(lyrics, *, topics = 15, top_words=20):
    """
    
    :param lyrics: 
    :return: 
    """
    logging.info("Finding Topics")
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF

    nmf_model = NMF(n_components=topics, random_state=1, alpha=.1, l1_ratio=.5)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=10, stop_words="english")

    data = []
    for s in tqdm(lyrics):
        data.append(s["text_raw"])
    tfidf = tfidf_vectorizer.fit_transform(data)
    result = nmf_model.fit_transform(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    topics = []
    for topic in tqdm(nmf_model.components_):
        topics.append([tfidf_feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]])

    for idx, song in enumerate(tqdm(lyrics)):
        song["topics"] = result[idx].tolist()

    return topics, lyrics


def save(lyrics, filename):
    """
    Saves the lyrics as a json dump to DIR_OUTPUT / filename 
    :param filename: The filename. .json is appended. 
    """
    filename += ".json"
    logging.info("Saving lyrics to {}".format(filename))
    outfile = DIR_OUTPUT / filename
    json_string = dumps(lyrics, indent=2)
    outfile.write_text(json_string)


if __name__ == '__main__':
    lyrics = loads(FILE_POP.read_text())
    lyrics = detect_language(lyrics)
    save(lyrics, "lan")
    lyrics = calculate_statistics(lyrics)
    save(lyrics, "lan_stats")
    lyrics = analyze_emotions(lyrics)
    save(lyrics, "lan_stats_emo")