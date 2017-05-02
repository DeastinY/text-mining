import json
import nltk
from nltk.stem.porter import PorterStemmer

INFILE = "lyrics_tokenized.json"
OUTFILE = "index.json"

porter_stemmer = PorterStemmer()

dictionary = dict()

"""
dictionary maps every token to a list of objects, where
each object in the list represents a song.
Each song has an id and a list of occurrences of the token
in the song given in the format (sentence_id, token_id).
Thus, the length of the token list is the document frequency
and the length of the places list the term frequency for
a certain song.

{
	token: [
		{
			song: id,
			places: [(sentence_id, token_id), (...), (...)]		
		},
		{...},
		{...}
	]
}
"""

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		for idx, song in enumerate(songs):
			for s_id, sentence in enumerate(song["text_tokenized"]):
				for t_id, token in enumerate(sentence):
					stemmed_token = porter_stemmer.stem(token.encode("ascii", "ignore"))

					if stemmed_token.isspace() or len(stemmed_token) == 0:
						continue

					if stemmed_token not in dictionary:
						dictionary[stemmed_token] = [{
							"song": song["id"],
							"places": [(s_id, t_id)]
						}]
					else:
						entry = filter(lambda x: x["song"] == song["id"], dictionary[stemmed_token])
						if len(entry) == 0:
							dictionary[stemmed_token].append({
								"song": song["id"],
								"places": [(s_id, t_id)]
							})
						elif len(entry) == 1:
							entry[0]["places"].append((s_id, t_id))
						else:
							print("Found an incorrect number of entries for song {} and token {}".format(song["id"], stemmed_token))

			print("Processed song {}".format(idx))
		json.dump(dictionary, outf, indent=2)