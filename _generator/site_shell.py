"""Shared Jinja environment for every generated page of the site (see templates/base.html)."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).parent
SITE = HERE.parent  # the personal-site repo this generator lives in
TEMPLATES = HERE / "templates"
FONTS = "https://fonts.googleapis.com/css2?family=Charis+SIL:ital,wght@0,400;0,700;1,400;1,700&display=swap"

env = Environment(
    loader=FileSystemLoader(TEMPLATES),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def root_for(out_rel):
    """Relative path from a page (given relative to the site root) back to the site root."""
    return "../" * (len(Path(out_rel).parts) - 1)


def render(template, out_rel, **context):
    """Render `template` as the page at `out_rel` (relative to the site root) and return the HTML."""
    return env.get_template(template).render(root=root_for(out_rel), fonts=FONTS, **context)
