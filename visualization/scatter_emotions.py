import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

INFILE = "data/results_per_artist.json"
categories = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
colors = ["red", "yellow", [106, 38, 158], "magenta", "green", "blue", [210, 222, 102], "orange"]

def main():
	with open(INFILE, "r") as inf:
		artists = json.load(inf)

	emotions_dict = list(artists.items())
	labels = [x for x, y in emotions_dict]
	emotions = [y["emotions"] for x, y in emotions_dict]

	model = TSNE(n_components=3, random_state=0)
	result = model.fit_transform(emotions)

	fig = plt.figure()
	ax = fig.gca(projection="3d")

	for i, vec in enumerate(result):
		ax.text(vec[0], vec[1], vec[2], labels[i], color=get_color(emotions[i]))

	fig.show()

def get_color(emotion):
	return colors[np.argmax(emotion)]


if __name__ == "__main__":
	main()