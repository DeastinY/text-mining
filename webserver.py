import logging
from pathlib import Path
from flask import Flask, render_template, request
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
DIR_WORDCLOUD = Path("output/wordclouds")


def get_artists():
    """Loads all artists that wordclouds are available for."""
    return [str(d).split('/')[-1] for d in DIR_WORDCLOUD.iterdir() if d.is_dir()]


@app.route("/", methods=["POST", "GET"])
def index():
    artists = get_artists()
    general, emotions = None, None
    if len(request.form) > 0:
        artist = request.form['artists']
        general = artist + "/general.png"
        emotions = artist + "/emotions.png"
    return render_template('index.html', artists=artists, general=general, emotions=emotions)

if __name__ == "__main__":
    app.run()