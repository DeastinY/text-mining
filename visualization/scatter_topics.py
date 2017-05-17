import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

INFILE = "data/results_per_artist.json"
colors = ["red", "yellow", np.array([106, 38, 158])/255, "magenta", "green", "blue", 
np.array([210, 222, 102])/255, "orange", "black", np.array([213, 241, 143])/255]

def main():
	with open(INFILE, "r") as inf:
		artists = json.load(inf)

	topics_dict = list(artists.items())
	labels = [x for x, y in topics_dict]
	topics = [y["topics"] for x, y in topics_dict]

	model = TSNE(n_components=3, random_state=0)
	result = model.fit_transform(topics)

	fig = plt.figure()
	ax = fig.gca(projection="3d")

	for i, vec in enumerate(result):
		ax.text(vec[0], vec[1], vec[2], labels[i], color=get_color(topics[i]))

	fig.show()

def get_color(topic):
	return colors[np.argmax(topic)]

if __name__ == "__main__":
	main()