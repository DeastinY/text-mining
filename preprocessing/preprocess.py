import json
import re

INFILE = "../lyric-crawler/lyrics_djent.txt"
OUTFILE = "lyrics.json"

"""
{
    artist: "",
    title: "",
    text_raw: "",
    text_tokenized: []
}
"""

results = []
current_song = dict()

with open(INFILE, "r") as inf:
    with open(OUTFILE, "w") as outf:
        for line in inf:
            if line.isspace():
                continue

            if line.startswith("$$$$"):
                # new artist
                print("New artist: " + line)
                current_artist = line.replace("$$$$$$$$$$", "").strip()
                current_song["artist"] = current_artist
            elif line.startswith("####"):
                # new title
                print("New title: " + line)
                if "text_raw" in current_song.keys():
                    results.append(current_song)
                    current_song = dict()
                    current_song["artist"] = current_artist
                current_song["title"] = line.replace("###########", "").strip()
                current_song["text_raw"] = ""
            else:
                current_song["text_raw"] += line
        json.dump(results, outf, indent=2)
