import json
from textstat.textstat import textstat

INFILE = "lyrics_with_sentiment.json"
OUTFILE = "lyrics_statistics.json"

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		for idx, song in enumerate(songs):
			song["num_syllables"] = textstat.syllable_count(song["text_raw"])
			song["num_words"] = textstat.lexicon_count(song["text_raw"])
			song["num_sentences"] = textstat.sentence_count(song["text_raw"])
			song["flesch_score"] = textstat.flesch_reading_ease(song["text_raw"])
			song["flesch_kincaid_level"] = textstat.flesch_kincaid_grade(song["text_raw"])
			song["fog_score"] = textstat.gunning_fog(song["text_raw"])
			song["num_difficult_words"] = textstat.dale_chall_readability_score(song["text_raw"])
			print("Processed song {}".format(idx))
		json.dump(songs, outf, indent=2)
