import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

INFILE = "lyrics_with_summarization.json"
OUTFILE = "lyrics_with_sentiment.json"

sid = SentimentIntensityAnalyzer()

with open(INFILE, "r") as inf:
    with open(OUTFILE, "w") as outf:
        songs = json.load(inf)
        for idx, song in enumerate(songs):
            song["sentiment"] = sid.polarity_scores(song["text_raw"])
            print("Processed song {}".format(idx))
        json.dump(songs, outf, indent=2)
