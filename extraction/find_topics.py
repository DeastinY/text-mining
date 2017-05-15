import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

if __name__ == "__main__":
	INFILE = "output/lyrics_emotions.json"
	OUTTOPICS = "output/topics.json"
	OUTSONGS = "output/lyrics_topics.json"

	#n_features = 1000
	n_topics = 15
	n_top_words = 20

	nmf_model = NMF(n_components=n_topics, random_state=1,
		alpha=.1, l1_ratio=.5)
	tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=10, stop_words="english")

	with open(INFILE, "r") as inf:
		songs = json.load(inf)
		data = []
		for idx, song in enumerate(songs):
			data.append(song["text_raw"])
		tfidf = tfidf_vectorizer.fit_transform(data)
		result = nmf_model.fit_transform(tfidf)
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