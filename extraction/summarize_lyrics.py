import json
from gensim.summarization import summarize

INFILE = "lyrics_with_keywords.json"
OUTFILE = "lyrics_with_summarization.json"

with open(INFILE, "r") as inf:
    with open(OUTFILE, "w") as outf:
        songs = json.load(inf)
        for idx, song in enumerate(songs):
            try:
                song["summarization"] = summarize(song["text_raw"])
                print("Processed song {}".format(idx))
            except TypeError:
                print("Skipped song {}".format(idx))
            except ValueError:
                print("Skipped song {}".format(idx))
        json.dump(songs, outf, indent=2)
