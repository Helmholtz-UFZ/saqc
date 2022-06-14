# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re
import subprocess

import markdown
import pytest
import requests
from bs4 import BeautifulSoup


@pytest.mark.parametrize("fname", ["README.md"])
def test_codeSnippets(fname):
    with open(fname, "r") as f:
        content = f.read()

    snippets = re.findall("`{3}python(.*)```", content, flags=re.M | re.DOTALL)

    for snippet in snippets:
        try:
            subprocess.check_output(["python", "-c", snippet])
        except subprocess.CalledProcessError:
            print(snippet)
            raise AssertionError("failed to execute code snippet")


FILEURL = "https://git.ufz.de/rdm-software/saqc/-/blob/develop/"


@pytest.mark.parametrize("fname", ["README.md"])
def test_links(fname):
    with open(fname, "r") as f:
        content = f.read()

    soup = BeautifulSoup(markdown.markdown(content), "html.parser")

    links = []
    # links
    for link in soup.find_all("a"):
        l = link.get("href").strip()
        if not l.startswith("mailto"):
            links.append(l)

    # images
    for link in soup.find_all("img"):
        links.append(link.get("src").strip())

    for link in links:
        if not link.startswith("http"):
            link = f"{FILEURL}/link"
        res = requests.get(link)
        if res.status_code != requests.codes.ok:
            raise AssertionError(f"failed to retrieve: {link}")
