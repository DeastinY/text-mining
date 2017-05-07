import json

INFILE = "stopwords.txt"
OUTFILE = "stopwords.json"

words = []

with open(INFILE, "r") as inf:
    with open(OUTFILE, "w") as outf:
        for line in inf:
            if len(line) == 0 or line.isspace():
                continue
            words.append(line.strip())
        json.dump(words, outf, indent=2)
