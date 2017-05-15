#!/usr/bin/env python3
import json
import csv
import nltk
import numpy as np

categories = ["anger", "anticipation", "disgust", "fear",
"joy", "sadness", "surprise", "trust"]

INFILE = "../lyric-crawler/lyrics.json"
OUTFILE = "output/lyrics_emotions.json"
EMOLEXFILE = "EmoLex.csv"

emolex = dict()
with open(EMOLEXFILE, "r") as ef:
	reader = csv.reader(ef, delimiter=";")

	header_row = next(reader)

	for row in reader:
		emolex[row[0].lower().strip()] = np.array(row[4:], dtype=int)

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		for idx, song in enumerate(songs):
			emotion_vector = np.zeros(8, dtype=int)
			sentences = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(song["text_raw"])]
			for sentence in sentences:
				for token in sentence:
					word = token.lower().strip()
					if word in emolex:
						emotion_vector += emolex[word]
			song["emotions"] = emotion_vector.tolist()
			print("Processed song {}".format(idx))
		json.dump(songs, outf, indent=2)