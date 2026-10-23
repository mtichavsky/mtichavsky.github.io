# Agent instructions

This repo is Milan Tichavský's personal site, served by GitHub Pages at mtichavsky.cz.
Everything committed here is public, including git history.

## No personal data on the site (GDPR)

- Never publish names or other personal data of individual people (players, contacts, anyone).
  The only exception is the site owner, Milan Tichavský.
- Team names are fine. Player names, line-ups, squads, per-player stats are not.
- Personal data may be kept on local disk for the generator (`_generator/data/`, git-ignored),
  but must never reach a rendered page, a committed file, a commit message or the git history.
- Before committing, check generated pages and code for people's names. If something slipped
  into a commit, tell the user: removing it requires rewriting history and a force push.

## Small football forecast

- `_generator/run.py` regenerates `maly-fotbal/`. It does not commit; the user reviews and pushes.
- Internal links point to files (`maly-fotbal/index.html`), not directories, so the site also
  works when opened locally from disk.

## Shared page shell (Jinja templates)

- `_generator/templates/base.html` is the one definition of the head, top bar and footer. Change
  them there, never in the generated HTML.
- Hand-written pages live in `_generator/templates/pages/` (e.g. `pages/index.html` -> `index.html`).
  Edit the template, then run `python _generator/build.py` to rewrite the published HTML. It also
  re-wraps the existing `maly-fotbal/` pages in the current shell without refetching data.
- `eeg/` is a standalone app with its own app bar and is not built from these templates.
