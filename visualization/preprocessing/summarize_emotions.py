import json
import numpy as np

INFILE = "../../extraction/lyrics_with_emotions.json"
OUTFILE = "../data/emotions_per_band.json"

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		summary = dict()
		counts = dict()
		for idx, song in enumerate(songs):
			if song["artist"] not in summary:
				summary[song["artist"]] = np.zeros(8, dtype=float)
				counts[song["artist"]] = 0

			summary[song["artist"]] += np.array(song["emotions"])
			counts[song["artist"]] += 1

			print("Processed song {}".format(idx))

		for k, v in summary.items():
			summary[k] = list(v/float(counts[k]))

		json.dump(summary, outf, indent=2)