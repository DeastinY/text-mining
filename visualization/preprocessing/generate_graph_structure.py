import json
import numpy as np

INFILE = "emotions_per_band.json"
OUTFILE = "link_network_data.json"

network = dict()
network["nodes"] = []
network["links"] = []

"""
with open(INFILE, "r") as inf:
	with open(OUTFILE, "w") as outf:
		bands = json.load(inf)
		for band, emotion_vector in bands.items():
			network["nodes"].append({
				"name": "",
				"artist": band,
				"id": ""
			})
"""