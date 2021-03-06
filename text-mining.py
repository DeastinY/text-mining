import logging
from json import loads, dumps
from pathlib import Path

import nltk
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
FILE_POP = Path("pop.json")
FILE_DB = Path("db.json")
FILE_DJENT = Path("djent.json")
FILE_EMOLEX = Path("EmoLex.csv")
FILE_STOPWORDS = Path("stopwords.json")
DIR_OUTPUT = Path("output")
EMOTION_CATEGORIES = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


### Text Processing


def tokenize(lyrics):
    logging.info("Tokenizing")
    import re
    from nltk.stem.porter import PorterStemmer
    regex = re.compile(r"[.:,;_')(`!?\-]")
    stemmer = PorterStemmer()
    for song in tqdm(lyrics):
        try:
            tokens = nltk.word_tokenize(song["text_raw"])
            song["text_tokenized"] = [stemmer.stem(token) for token in tokens if len(regex.findall(token)) == 0]
        except Exception as e:
            logging.error("Something bad happened in the current song ! Skipping it... \n{}".format(song))
            logging.exception(e)
    return lyrics


def analyze_emotions(lyrics, *, emolex=None):
    """
    Analyzes emotions based on EmoLex. Annotates the passed lyrics.
    :param lyrics: The lyrics.
    :param emolex: The EmoLex. If empty executes read_emolex.
    :return: Annotated lyrics
    """
    logging.info("Analyzing Emotions")
    emolex = read_emolex() if not emolex else emolex
    for idx, song in tqdm(enumerate(lyrics), total=len(lyrics)):
        try:
            if not "language" in song:
                logging.warning("Song is not annotated with language. Execute detect_language first !")
                continue
            if song["language"] != "en":
                logging.debug("Skipping {} as it's not English.".format(song["title"]))
                continue

            emotion_vector = np.zeros(8, dtype=int)
            sentences = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(song["text_raw"])]
            for sen in sentences:
                for t in sen:
                    word = t.lower().strip()
                    if word in emolex:
                        emotion_vector += emolex[word]
            song["emotions"] = emotion_vector.tolist()
        except Exception as e:
            logging.error("Something bad happened in the current song ! Skipping it... \n{}".format(song))
            logging.exception(e)
    return lyrics


def calculate_statistics(lyrics):
    """
    Calculates statistics based on the text_raw of the lyrics.
    :return: Annotated lyrics containing information about the songs
    """
    logging.info("Calculating Statistics")
    from textstat.textstat import textstat
    for idx, song in tqdm(enumerate(lyrics), total=len(lyrics)):
        try:
            song["num_syllables"] = textstat.syllable_count(song["text_raw"])
            song["num_words"] = textstat.lexicon_count(song["text_raw"])
            song["num_sentences"] = textstat.sentence_count(song["text_raw"])
            song["flesch_score"] = textstat.flesch_reading_ease(song["text_raw"])
            song["flesch_kincaid_level"] = textstat.flesch_kincaid_grade(song["text_raw"])
            song["fog_score"] = textstat.gunning_fog(song["text_raw"])
            song["num_difficult_words"] = textstat.dale_chall_readability_score(song["text_raw"])
        except Exception as e:
            logging.error("Something bad happened in the current song ! Skipping it... \n{}".format(song))
            logging.exception(e)
    return lyrics


def detect_language(lyrics):
    """
    Detects a song's language. Always assigns the most probable language tag, even if the text is utter nonsense.
    :return: 
    """
    logging.info("Detecting language")
    from langdetect import detect
    for song in tqdm(lyrics):
        try:
            song["language"] = detect(song["text_raw"])
        except Exception:
            logging.error("Could not detect language for the song. Setting to ?... \n{}".format(song))
            song["language"] = "?"
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
        try:
            r.extract_keywords_from_text(song["text_raw"])
            song["keywords"] = r.get_ranked_phrases()[:top_keywords]
        except Exception as e:
            logging.error("Something bad happened in the current song ! Skipping it... \n{}".format(song))
            logging.exception(e)
    return lyrics


def find_topics(lyrics, *, features=3000, topics=10, top_words=20):
    """
    Finds topics and annotates them to the lyrics.
    
    
    :param lyrics: The lyrics to work on. Need to have the language tag, will only consider "en" songs.
    :param features: How many features to consider.
    :param topics: How many topics to generate.
    :param top_words: How many words to return per topic. 
    :return: Annotated lyrics and a list of topics.
    """
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    lyrics = np.array(lyrics)
    additional_stopwords = loads(FILE_STOPWORDS.read_text())

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if len(regex.findall(token)) == 0]
        try:
            return [stemmer.stem(token) for token in tokens]
        except IndexError:  # No idea where this comes from TODO: Investigate
            return tokens

    regex = re.compile(r"[.:,;-_')(`!?]")
    stemmer = PorterStemmer()
    stopset = set(stopwords.words())
    stopset = stopset.union(additional_stopwords)

    nmf_model = NMF(n_components=topics, random_state=1, alpha=.1, l1_ratio=.5)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.75, max_features=features, strip_accents="ascii",
                                       analyzer="word", stop_words=list(stopset))  # TODO: Add custom tokenizer again
    english_indices = np.where(np.array([song["language"] for song in lyrics]) == "en")[0]
    data = [song["text_raw"] for song in lyrics[english_indices]]
    logging.info("Building TF_IDF features")
    tfidf = tfidf_vectorizer.fit_transform(data)
    logging.info("Fitting NMF model")
    result = nmf_model.fit_transform(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    topics = []
    for topic in nmf_model.components_:
        topics.append([tfidf_feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]])

    for idx, topic_vec in enumerate(result):
        lyrics[english_indices[idx]]["topics"] = topic_vec.tolist()

    return lyrics.tolist(), topics


### Visualization


def generate_wordclouds(lyrics, missing_only=True):
    """
    Generates wordclouds for artists songs and the emotions
    :param lyrics: The lyrics to work on, annotated with emotions. Saves the results to OUTPUT_DIR/WORDCLOUDS.
    :return: Nothing. Saves to OUTPUT_DIR/wordclouds.
    """
    logging.info("Building wordclouds")
    from wordcloud import WordCloud
    wc = WordCloud(scale=2, stopwords=loads(FILE_STOPWORDS.read_text()))
    wordclouds = {}
    artist_texts = {}
    artist_emotions = {}
    for song in lyrics:
        try:
            if not "emotions" in song:
                logging.debug("Skipping song as no emotion information is available. {}".format(song["title"]))
                continue
            title, artist, text, emotions = song["title"], song["artist"], song["text_raw"], song["emotions"]
            if artist not in wordclouds:
                wordclouds[artist] = {}
                artist_texts[artist] = []
                artist_emotions[artist] = []
            else:
                artist_texts[artist].append(text)
                # Add the most significant emotion per song
                artist_emotions[artist].append(EMOTION_CATEGORIES[emotions.index(max(emotions))])
                #wordclouds[artist][title] = wc.generate(text)
        except Exception as e:
            logging.error("Something bad happened in the current song ! Skipping it... \n{}".format(song))
            logging.exception(e)

    outdir = DIR_OUTPUT / "wordclouds"
    outdir.mkdir(exist_ok=True)
    logging.info("Generating artist wordclouds")
    for artist, text in tqdm(artist_texts.items()):
        try:
            outfile = outdir / str(artist).strip().replace(' ', '_').replace('/', '')
            if missing_only and (outfile/"general.png").is_file():
                continue
            else:
                outfile.mkdir(exist_ok=True)
            if len(text) != 0:
                cloud = wc.generate(" ".join(text))
                cloud.to_file(str(outfile / "general.png"))
        except ZeroDivisionError as e:  # TODO: Investigate why this happens
            logging.error("ZeroDivisionError on text for {}".format(artist))
    logging.info("Generating artist emotion wordclouds")
    for artist, emotions in tqdm(artist_emotions.items()):
        try:
            outfile = outdir / str(artist).strip().replace(' ', '_').replace('/', '')
            if missing_only and (outfile/"emotions.png").is_file():
                continue
            else:
                outfile.mkdir(exist_ok=True)
            if len(text) != 0:
                cloud = wc.generate(" ".join(emotions))
                cloud.to_file(str(outfile / "emotions.png"))
        except ZeroDivisionError as e: # TODO: Investigate why this happens
            logging.error("ZeroDivisionError on emotions for {}".format(artist))
    logging.info("Saving wordclouds")


### Not Implemented


def build_index(lyrics):
    """
        Builds a TF-IDF index.
        :param lyrics: The lyrics.
        :return: The index.
        """
    logging.info("Grabbing Lyrics")
    logging.warning("NOT IMPLEMENTED")
    ...


def get_lyrics(source):
    """
    Grabs lyrics either based on the Bilboard Top 100 or got-djent.com.
    :param source: Either "pop" or "djent"
    :return: The crawled lyrics.
    """
    logging.info("Grabbing Lyrics")
    logging.warning("NOT IMPLEMENTED")
    ...


### Util

def read_database():
    """Generates a new json file from the db.sqlite database."""
    import sqlite3
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    raw_data = [r for r in c.execute(r"SELECT A.name, S.name, S.lyric FROM artistSongs B JOIN artist A on B.artistId = A.id JOIN songs S on B.songId = S.id;")]
    lyrics = []
    for r in raw_data:
        lyrics.append({
            "artist": r[0],
            "title": r[1],
            "text_raw": r[2]
        })
    return lyrics


def read_emolex():
    """
    Reads the EmoLex csv file.
    :return: A dictionary with the contents.
    """
    logging.info("Reading {}".format(FILE_EMOLEX))
    emolex = {}
    import csv
    with FILE_EMOLEX.open("r") as ef:
        reader = csv.reader(ef, delimiter=';')
        next(reader)  # skip header
        for row in tqdm(reader):
            emolex[row[0].lower().strip()] = np.array(row[4:], dtype=int)
    return emolex


def save(lyrics, filename):
    """
    Saves the lyrics as a json dump to DIR_OUTPUT / filename 
    :param filename: The filename. .json is appended. 
    """
    filename += ".json"
    logging.info("Saving to {}".format(filename))
    outfile = DIR_OUTPUT / filename
    json_string = dumps(lyrics, indent=2)
    outfile.write_text(json_string)


def merge_json():
    """
    Utility function that merges all mergeable (= raw_text field and no collisions) json files in the output directory.
    Can be used to hack multiprocessing by simply running the program with different modes in parallel and merging later. 
    :return: A merged json object
    """
    logging.info("Merging JSON files")
    merged = []  # Assume a list of lyric-dicts
    for f in DIR_OUTPUT.iterdir():
        error = False
        if not f.is_file() or not '.json' in str(f):
            logging.warning("Skipping file {}".format(f))
            continue
        logging.info("Loading {}".format(f))
        text = f.read_text()
        lyrics = loads(text)
        if not isinstance(lyrics, list):
            logging.warning("Skipping file {}".format(f))
            continue
        for idx, song in enumerate(lyrics):
            if not isinstance(song, dict):
                logging.warning("Skipping file {}".format(f))
                break
            if len(merged) == 0:  # No need to merge with the first lyrics file
                merged = lyrics
            for key in song.keys():
                if not key in merged[idx]:
                    merged[idx][key] = song[key]


    return merged


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="Can be language, stats, emotions, keywords, topics, wordclouds.")
    args = parser.parse_args()

    file_merged = Path("output/db_merged.json")
    logging.info("Loading file")
    lyrics = loads(file_merged.read_text())
    prefix = "db_"

    if args.mode == 'create':
        db = read_database()
        logging.info("Saving database to db.json")
        Path("db.json").write_text(dumps(db))
    elif args.mode == 'language':
        lyrics = detect_language(lyrics)
        save(lyrics, prefix + "language")
    elif args.mode == 'stats':
        lyrics = calculate_statistics(lyrics)
        save(lyrics, prefix + "stats")
    elif args.mode == 'emotions':
        lyrics = analyze_emotions(lyrics)
        save(lyrics, prefix + "emotions")
    elif args.mode == 'keywords':
        lyrics = extract_keywords(lyrics)
        save(lyrics, prefix + "keywords")
    elif args.mode == 'topics':
        lyrics, topics = find_topics(lyrics)
        save(topics, prefix + "just_topics")
        save(lyrics, prefix + "topics")
    elif args.mode == 'merge':
        merged = merge_json()
        save(merged, prefix + "merged")
    elif args.mode == 'wordclouds':
        wordclouds = generate_wordclouds(lyrics)
    else:
        logging.error("Did not understand mode. Must be language, stats, emotion, keywords, topics, wordclouds")
