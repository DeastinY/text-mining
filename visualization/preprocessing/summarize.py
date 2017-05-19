import json
import numpy as np

INFILE = "../../output/bill_lyrics_annotated.json"
OUTFILE = "../data/result.graphml"


def summarize_songs(songs):
	from numpy.linalg import norm
	summary = dict()

	english_indices = np.where(np.array([song["language"] for song in songs]) == "en")[0]
	songs = np.array(songs)[english_indices]

	for idx, song in enumerate(songs):
		if song["artist"] not in summary:
			summary[song["artist"]] = {
				"emotions": np.zeros(8, dtype=float),
				"topics": np.zeros(len(song["topics"]), dtype=float)
			}

		summary[song["artist"]]["emotions"] += np.array(song["emotions"])
		summary[song["artist"]]["topics"] += np.array(song["topics"])

	for artist, value in summary.items():
		summary[artist]["emotions"] = (summary[artist]["emotions"]/norm(summary[artist]["emotions"])).tolist()
		summary[artist]["topics"] = (summary[artist]["topics"]/norm(summary[artist]["topics"])).tolist()

	return summary

def calculate_similarity(summary):
	from scipy.spatial.distance import cdist
	items = summary.items()
	emotions = np.array([obj["emotions"] for artist, obj in items])
	topics = np.array([obj["topics"] for artist, obj in items])

	similarity = {}

	for artist, obj in items:
		similarity[artist] = {}
		similarity[artist]["emotion_dist"] = cdist(np.array([obj["emotions"]]), emotions, metric="cosine")[0]
		#similarity[artist]["emotion_dist"] /= np.max(similarity[artist]["emotion_dist"])
		similarity[artist]["topic_dist"] = cdist(np.array([obj["topics"]]), topics, metric="cosine")[0]
		#similarity[artist]["topic_dist"] /= np.max(similarity[artist]["topic_dist"])

	return similarity

def generate_graph(similarity, *, emotion_threshold=0.02, topic_threshold=0.3):
	from pygraphml import Graph
	items = similarity.items()
	labels = np.array([artist for artist, obj in items])

	g = Graph()

	for artist in labels:
		g.add_node(artist)

	for artist_id, x in enumerate(items):
		artist, obj = x

		for idx, emotion in enumerate(obj["emotion_dist"]):
			if 0 < emotion <= emotion_threshold:
				g.add_edge_by_label(labels[artist_id], labels[idx])

		"""
		for idx, topic in enumerate(obj["topic_dist"]):
			if topic <= topic_threshold:
				g.add_edge_by_label(labels[artist_id], labels[idx])
		"""

	return g

if __name__ == "__main__":
	with open(INFILE, "r") as inf:
		songs = json.load(inf)

	summary = summarize_songs(songs)
	similarity = calculate_similarity(summary)
	graph = generate_graph(similarity)

	from pygraphml import GraphMLParser
	parser = GraphMLParser()
	parser.write(graph, OUTFILE)