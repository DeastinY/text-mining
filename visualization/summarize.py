import json
import numpy as np

INFILE = "../output/bill_lyrics_annotated.json"
OUTEMOTIONS = "data/emotion_graph.graphml"
OUTTOPICS = "data/topic_graph.graphml"
OUTEMOTIONSCENTERS = "data/emotion_graph_clusters.graphml"
OUTTOPICCENTERS = "data/topic_graph_clusters.graphml"

EMOTION_CATEGORIES = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
TOPIC_CATEGORIES = ["time", "bitch, fuck 'n money", "love", "girl & boy", "crazy baby", "breakup", "party", "young & rich", "good & bad", "feel real"]

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
	from sklearn.metrics.pairwise import cosine_similarity
	items = summary.items()
	emotions = np.array([obj["emotions"] for artist, obj in items])
	topics = np.array([obj["topics"] for artist, obj in items])

	emotion_scores = cosine_similarity(emotions)
	topic_scores = cosine_similarity(topics)

	similarity = {}

	for idx, x in enumerate(items):
		artist, obj = x
		similarity[artist] = {}
		similarity[artist]["emotion_sim"] = emotion_scores[idx]
		similarity[artist]["topic_sim"] = topic_scores[idx]

	return similarity

def generate_graph(similarity, attr):
	from pygraphml import Graph
	items = similarity.items()
	labels = np.array([artist for artist, obj in items])

	network = np.zeros((labels.size, labels.size))

	g = Graph()

	for artist in labels:
		g.add_node(artist)

	for artist_id, x in enumerate(items):
		network[artist_id, artist_id] = 1
		artist, obj = x

		if attr == "emotion":
			for idx, score in enumerate(obj["emotion_sim"]):
				if network[artist_id, idx] == 0 and network[idx, artist_id] == 0:
					edge = g.add_edge_by_label(labels[artist_id], labels[idx])
					edge["weight"] = score
					network[artist_id, idx] = 1
					network[idx, artist_id] = 1
		elif attr == "topic":
			for idx, score in enumerate(obj["topic_sim"]):
				if network[artist_id, idx] == 0 and network[idx, artist_id] == 0:
					edge = g.add_edge_by_label(labels[artist_id], labels[idx])
					edge["weight"] = score
					network[artist_id, idx] = 1
					network[idx, artist_id] = 1

	return g

def generate_graph_with_clusters(summary, attr):
	from pygraphml import Graph
	items = summary.items()
	labels = np.array([artist for artist, obj in items])

	g = Graph()

	for artist in labels:
		g.add_node(artist)

	if attr == "emotion":
		for category in EMOTION_CATEGORIES:
			g.add_node(category)
	elif attr == "topic":
		for category in TOPIC_CATEGORIES:
			g.add_node(category)

	for artist_id, x in enumerate(items):
		arist, obj = x

		if attr == "emotion":
			for idx, score in enumerate(obj["emotions"]):
				edge = g.add_edge_by_label(EMOTION_CATEGORIES[idx], labels[artist_id])
				if score == 0:
					edge["weight"] = 0.001 # Set to very small value
				else:
					edge["weight"] = score

		elif attr == "topic":
			for idx, score in enumerate(obj["topics"]):
				edge = g.add_edge_by_label(TOPIC_CATEGORIES[idx], labels[artist_id])
				if score == 0:
					edge["weight"] = 0.001 # Set to very small value
				else:
					edge["weight"] = score

	return g

if __name__ == "__main__":
	with open(INFILE, "r") as inf:
		songs = json.load(inf)

	summary = summarize_songs(songs)
	similarity = calculate_similarity(summary)
	emotion_graph = generate_graph(similarity, "emotion")
	topic_graph = generate_graph(similarity, "topic")
	emotion_graph_with_clusters = generate_graph_with_clusters(summary, "emotion")
	topic_graph_with_clusters = generate_graph_with_clusters(summary, "topic")

	from pygraphml import GraphMLParser
	parser = GraphMLParser()
	parser.write(emotion_graph, OUTEMOTIONS)
	parser.write(topic_graph, OUTTOPICS)

	# Need to manually change
	# <key attr.name="weight" attr.type="string" id="weight"/>
	# to
	# <key attr.name="weight" attr.type="double" for="edge" id="weight"/>
	# in output files
	parser.write(emotion_graph_with_clusters, OUTEMOTIONSCENTERS)
	parser.write(topic_graph_with_clusters, OUTTOPICCENTERS)