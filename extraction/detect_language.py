import json
import enchant
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("output")
INFILE = OUTPUT / "lyrics_emotions.json"
OUTFILE = OUTPUT / "lyrics_with_emotions_with_language_we_need_a_new_naming_sheme.json"
THRESHOLD = 0.6

with INFILE.open("r") as inf:
    with OUTFILE.open("w") as outf:
        songs = json.load(inf)
        d = enchant.Dict("en_US")
        for idx, song in tqdm(enumerate(songs), total=len(songs)):
            ratings = [d.check(w) for w in song["text_raw"]]
            trues, falses = ratings.count(True), ratings.count(False)
            ratio = trues / (trues + falses)
            song["language"] = "en_US" if ratio > THRESHOLD else ""
        json.dump(songs, outf, indent=2)
