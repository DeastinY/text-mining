import json
import requests
import pylyrics3

from bs4 import BeautifulSoup

def get_top():
    """Grabs the top 100 billboard artist"""
    RANKING_URL = "http://www.billboard.com/charts/artist-100"
    data = requests.get(RANKING_URL)
    soup = BeautifulSoup(data.text, "lxml")
    names = []
    for row in soup.find_all('h2'):
        if row.has_attr('class') and "chart-row__song" in row.get('class'):
            names.append(row.text)
    return names

def get_top_djent(num):
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

def get_lyrics(band):
    """Get the lyrics for all songs of a band."""
    return pylyrics3.get_artist_lyrics(band)

if __name__ == '__main__':
    with open('lyrics.json', 'w') as fout:
        all_lyrics = []
        for t in get_top_djent(300):
            print("Extracting {}".format(t))
            lyrics = get_lyrics(t)
            if lyrics and len(lyrics) > 0:
                band = t
                for song, lyrics in lyrics.items():
                    all_lyrics.append({
                        "text_raw": lyrics,
                        "interpret": band,
                        "title": song
                    })
        json.dump(all_lyrics, fout)
