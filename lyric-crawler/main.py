import requests
import pylyrics3

from bs4 import BeautifulSoup

RANKING_URL = "http://got-djent.com/bands/ranking?page="

def get_top(num):
    """Grabs the top num Djent bands by popularity from got-djent.com"""
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
    with open('lyrics.txt', 'w') as fout:
        for t in get_top(250):
            lyrics = get_lyrics(t)
            if lyrics and len(lyrics) > 0:
                fout.writelines('\n$$$$$$$$$$ {} $$$$$$$$$$\n'.format(t))
                for key, value in lyrics.items():
                    title = '\n########### {} ###########\n'.format(key)
                    fout.writelines([title, value])
