import json
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

def tokenize(text):
	tokens = nltk.word_tokenize(text)
	return [stemmer.stem(token) for token in tokens if len(regex.findall(token)) == 0]

if __name__ == "__main__":
	INFILE = "output/lyrics_emotions.json"
	INSTOPWORDSFILE = "stopwords.json"
	OUTTOPICS = "output/topics.json"
	OUTSONGS = "output/lyrics_topics.json"

	with open(INSTOPWORDSFILE, "r") as sf:
		additional_stopwords = json.load(sf)

	n_features = 3000
	n_topics = 10
	n_top_words = 20

	regex = re.compile(r"[.:,;-_')(`!?]")
	stemmer = PorterStemmer()
	stopset = set(stopwords.words())
	stopset = stopset.union(additional_stopwords)

	nmf_model = NMF(n_components=n_topics, random_state=1,
		alpha=.1, l1_ratio=.5)
	tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.75, max_features=n_features,
	strip_accents="ascii", analyzer="word", stop_words=list(stopset))

	with open(INFILE, "r") as inf:
		songs = json.load(inf)
		data = []
		for idx, song in enumerate(songs):
			data.append(song["text_raw"])
		tfidf = tfidf_vectorizer.fit_transform(data)
		print("Built tf-idf features")

		result = nmf_model.fit_transform(tfidf)
		print("Fit NMF model")

		tfidf_feature_names = tfidf_vectorizer.get_feature_names()
		topics = []
		for topic in nmf_model.components_:
			topics.append([tfidf_feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])

		for idx, song in enumerate(songs):
			song["topics"] = result[idx].tolist()

	with open(OUTTOPICS, "w") as outf:
		json.dump(topics, outf, indent=2)

	with open(OUTSONGS, "w") as outf:
		json.dump(songs, outf, indent=2)