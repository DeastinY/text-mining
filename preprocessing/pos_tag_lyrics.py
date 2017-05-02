import json
import nltk

"""
Extend object to

{
    interpret: "",
    title: "",
    text_raw: "",
    text_tokenized: [],
    text_pos_tagged: []
}
"""

INFILE = "lyrics_tokenized.json"
OUTFILE = "lyrics_pos_tagged.json"

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		for idx, song in enumerate(songs):
			song["text_pos_tagged"] = [nltk.pos_tag(sent) for sent in song["text_tokenized"]]
			print("Processed song {}".format(idx))
		json.dump(songs, outf, indent=2)