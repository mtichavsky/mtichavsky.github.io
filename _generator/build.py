"""Rebuild the site's pages from templates/ after editing the shared shell or a page.

    python _generator/build.py

- templates/pages/<path>.html  ->  <site>/<path>.html   (home, Messenger, ...)
- maly-fotbal/ pages are re-wrapped in the current shell, keeping their generated content, so a
  change to the top bar or footer reaches them without refetching data (run.py does a full refresh).

The EEG app (eeg/) is a standalone app with its own bar and is not built from these templates.
Nothing is committed: review and push the site repo yourself.
"""
import re
from html import unescape

import report
import site_shell as shell

PAGES = shell.TEMPLATES / "pages"
FOOTBALL = ["maly-fotbal/index.html", "maly-fotbal/model/index.html"]


def build_pages():
    for src in sorted(PAGES.rglob("*.html")):
        rel = src.relative_to(PAGES).as_posix()
        (shell.SITE / rel).write_text(shell.render(f"pages/{rel}", rel), encoding="utf-8")
        yield rel


def rewrap_football():
    for rel in FOOTBALL:
        out = shell.SITE / rel
        if not out.exists():
            continue
        html = out.read_text(encoding="utf-8")
        title = unescape(re.search(r"<title>(.*?)</title>", html, re.S)[1])
        description = unescape(re.search(r'<meta name="description" content="(.*?)"', html, re.S)[1])
        body = re.search(r"<main>\n(.*)\n  </main>", html, re.S)[1]
        out.write_text(report.page(title, description, body, shell.root_for(rel)), encoding="utf-8")
        yield rel


if __name__ == "__main__":
    for rel in [*build_pages(), *rewrap_football()]:
        print(f"wrote {rel}")
