import json
import requests
import pylyrics3
import os

from bs4 import BeautifulSoup

def get_top_billboard_artists():
    """Grabs the top 100 billboard artists"""
    RANKING_URL = "http://www.billboard.com/charts/artist-100"
    data = requests.get(RANKING_URL)
    soup = BeautifulSoup(data.text, "lxml")
    names = []
    for row in soup.find_all('h2'):
        if row.has_attr('class') and "chart-row__song" in row.get('class'):
            names.append(row.text)
    return names

def get_top_djent_artists(num):
    """Grabs the top num Djent bands by popularity from got-djent.com"""
    RANKING_URL = "http://got-djent.com/bands/ranking?page="
    def get_top_rec(page):
        """Recursively grabs the top num Djent bands by popularity from got-djent.com"""
        names = []
        data = requests.get(RANKING_URL+str(page)).text
        soup = BeautifulSoup(data, 'lxml')
        for row in soup.find_all('li'):
            if row.has_attr('class') and 'views-row' in row.get('class'):
                for name in row.find_all('a'):
                    name = name.get('href')
                    if 'band' in name:
                        name = name.split('/')[-1]
                        name = name.replace('-', ' ')
                        names.append(name)
        return set(names)

    names = []
    for i in range((num//20)+1):
        names.extend(get_top_rec(i))
    return names[0:num]

def get_songs(artist):
    result = []
    print("Extracting {}".format(artist))
    songs = pylyrics3.get_artist_lyrics(artist)
    if songs and len(songs) > 0:
        for song, lyrics in songs.items():
            result.append({
                "text_raw": lyrics,
                "artist": artist,
                "title": song
            })
    return result

if __name__ == '__main__':
    OUTFILE = "lyrics.json"
    with open(OUTFILE, "a") as fout:
        try:
            result = json.load(fout)
        except:
            result = []

        existing_artists = set([song["artist"] for song in result])
        for idx, artist in enumerate(get_top_billboard_artists()):
            if artist in existing_artists:
                print("Skipping {}".format(artist))
                continue

            result += get_songs(artist)

        for idx, artist in enumerate(get_top_djent_artists(50)):
            if artist in existing_artists:
                print("Skipping {}".format(artist))
                continue

            result += get_songs(artist)

        json.dump(result, fout, indent=2)
