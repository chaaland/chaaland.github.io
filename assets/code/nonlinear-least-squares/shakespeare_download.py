import os

import requests
from bs4 import BeautifulSoup

pjoin = os.path.join


def download_text(url: str) -> str:
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, "html.parser")
    text = soup.find_all(text=True)

    return text


if __name__ == "__main__":
    if "hamlet.txt" not in os.listdir(pjoin("..", "txt")):
        hamlet_url = "http://shakespeare.mit.edu/hamlet/full.html"
        hamlet_text = download_text(hamlet_url)
        output = ""
        for t in hamlet_text:
            if t.parent.name.lower() == "a":
                output += f"{t} "

        with open(pjoin("..", "..", "txt", "hamlet.txt"), "wt") as f:
            f.writelines(output)
