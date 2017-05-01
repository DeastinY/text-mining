import json
import nltk

INFILE = "lyrics.json"
OUTFILE = "lyrics_tokenized.json"

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		for idx, song in enumerate(songs):
			song["text_tokenized"] = [nltk.word_tokenize(line) for line in song["text_raw"].split("\n")]
			song["id"] = idx
			print("Processed song {}".format(idx))
		json.dump(songs, outf, indent=2)