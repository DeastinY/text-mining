import json
import nltk
from nltk.stem.porter import PorterStemmer

INFILE = "lyrics_tokenized.json"
OUTFILE = "index.json"

porter_stemmer = PorterStemmer()

dictionary = dict()

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		for idx, song in enumerate(songs):
			for sentence in song["text_tokenized"]:
				for token in sentence:
					stemmed_token = porter_stemmer.stem(token.encode("ascii", "ignore"))

					if stemmed_token.isspace() or len(stemmed_token) == 0:
						continue

					if stemmed_token not in dictionary:
						dictionary[stemmed_token] = [song["id"]]
					else:
						if song["id"] not in dictionary[stemmed_token]:
							dictionary[stemmed_token].append(song["id"])

			print("Processed song {}".format(idx))
		json.dump(dictionary, outf, indent=2)