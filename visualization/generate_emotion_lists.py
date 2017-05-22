import json
import numpy as np

INFILE = "../data/results_per_artist.json"

def main():
	with open(INFILE, "r") as inf:
		artists = json.load(inf)

	emotion_dict = list(artists.items())
	labels = np.array([x for x, y in emotion_dict])
	emotions = [y["emotions"] for x, y in emotion_dict]

	result = {
	"anger": get_emotion(0, labels, emotions),
	"anticipation": get_emotion(1, labels, emotions),
	"disgust": get_emotion(2, labels, emotions),
	"fear": get_emotion(3, labels, emotions),
	"joy": get_emotion(4, labels, emotions),
	"sadness": get_emotion(5, labels, emotions),
	"surprise": get_emotion(6, labels, emotions),
	"trust": get_emotion(7, labels, emotions)
	}
	
	for k, v in result.items():
		with open("../data/ranking_{}.json".format(k), "w") as outf:
			json.dump(v, outf, indent=2)

def get_emotion(idx, labels, emotions):
	return [(artist, value) for artist, value in zip(labels[np.argsort([x[idx] for x in emotions])], np.sort([x[idx] for x in emotions]))][::-1]

if __name__ == "__main__":
	main()