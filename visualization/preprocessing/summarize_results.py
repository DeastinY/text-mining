import json
import numpy as np
from numpy.linalg import norm

"""
Aggregates emotion and topic vectors for each artist.
"""

INFILE = "../../extraction/output/lyrics_topics.json"
OUTFILE = "../data/results_per_artist.json"

with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		songs = json.load(inf)
		summary = dict()
		counts = dict()
		for idx, song in enumerate(songs):
			if song["artist"] not in summary:
				summary[song["artist"]] = {
					"emotions": np.zeros(8, dtype=float),
					"topics": np.zeros(len(song["topics"]), dtype=float)
				}
				counts[song["artist"]] = 0

			summary[song["artist"]]["emotions"] += np.array(song["emotions"])
			summary[song["artist"]]["topics"] += np.array(song["topics"])
			counts[song["artist"]] += 1

			print("Processed song {}".format(idx))

		for artist, value in summary.items():
			# Not sure if it's better to normalize vectors or just
			# divide by the number of songs per artist
			summary[artist]["emotions"] = (summary[artist]["emotions"]/norm(summary[artist]["emotions"])).tolist()
			summary[artist]["topics"] = (summary[artist]["topics"]/norm(summary[artist]["topics"])).tolist()

		json.dump(summary, outf, indent=2)