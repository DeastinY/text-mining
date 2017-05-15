from rake_nltk import Rake
import json

INFILE = "../preprocessing/lyrics.json"
OUTFILE = "lyrics_with_keywords.json"

r = Rake()
n_top_keywords = 3

with open(INFILE, "r") as inf:
    with open(OUTFILE, "w") as outf:
        songs = json.load(inf)
        for idx, song in enumerate(songs):
            r.extract_keywords_from_text(song["text_raw"])
            song["keywords"] = r.get_ranked_phrases()[:n_top_keywords]
        json.dump(songs, outf, indent=2)