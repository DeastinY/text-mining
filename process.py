import json
from informationminer import InformationMiner

if __name__ == '__main__':
    INPUT = "preprocessing/lyrics.json"
    with open(INPUT, 'r') as fin:
        lyrics = json.load(fin)
    im = InformationMiner([l["text_raw"] for l in lyrics], save_output=True)
