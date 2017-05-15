import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

def print_top_words(model, feature_names, n_top_words):
	for idx, topic in enumerate(model.components_):
		print("Topic {}".format(idx))
		print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
		print()

if __name__ == "__main__":
	INFILE = "lyrics_statistics.json"
	#OUTFILE = ""

	n_features = 1000
	n_topics = 20
	n_top_words = 20

	nmf_model = NMF(n_components=n_topics, random_state=1,
		alpha=.1, l1_ratio=.5)
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
		max_features=n_features, stop_words="english")

	with open(INFILE, "r") as inf:
		#with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		data = []
		for idx, song in enumerate(songs):
			data.append(song["text_raw"])
		tfidf = tfidf_vectorizer.fit_transform(data)
		nmf_model.fit(tfidf)
		tfidf_feature_names = tfidf_vectorizer.get_feature_names()
		print_top_words(nmf_model, tfidf_feature_names, n_top_words)