import json
import nltk
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve

def get_features(songs):
	import re
	from nltk.corpus import stopwords
	from nltk.stem.porter import PorterStemmer
	from sklearn.feature_extraction.text import TfidfVectorizer

	lyrics = [song["text_raw"] for song in songs]

	with open("../stopwords.json") as sf:
		additional_stopwords = json.load(sf)

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

	tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.75, max_features=3000, strip_accents="ascii",
	                                   analyzer="word", stop_words=list(stopset))

	print("Building tf-idf features")
	features = tfidf_vectorizer.fit_transform(lyrics)
	print("tf-idf features built")

	return features.todense()

def get_labels(songs):
	return np.array([np.argmax(song["topics"]) for song in songs])

def split_dataset(features, labels, topic, test_size):
	from sklearn.model_selection import train_test_split

	positives = np.where(labels == topic)[0]
	print("Number of positive samples: {}".format(positives.size))
	negatives = np.where(labels != topic)[0]
	print("Number of negative samples: {}".format(negatives.size))
	ret = labels.copy()
	ret[positives] = 1
	ret[negatives] = -1

	return train_test_split(features, ret, test_size=test_size, random_state=0)

if __name__ == "__main__":
	INFILE = "../output/bill_lyrics_annotated.json"

	with open(INFILE, "r") as inf:
		songs = json.load(inf)

	english_indices = np.where(np.array([song["language"] for song in songs]) == "en")[0]
	english_songs = np.array(songs)[english_indices]

	labels = get_labels(english_songs)
	features = get_features(english_songs)
	train_data, test_data, train_labels, test_labels = split_dataset(features, labels, topic=0, test_size=0.2)

	svm = SVC()
	svm.fit(train_data, train_labels)

	predictions = svm.predict(test_data)
	precision, recall, _ = precision_recall_curve(test_labels, predictions)
	print("Precision: {}".format(precision))
	print("Recall: {}".format(recall))