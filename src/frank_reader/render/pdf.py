from io import BytesIO
from pathlib import Path

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# xhtml2pdf/reportlab's built-in base fonts (Helvetica etc.) have no Cyrillic
# glyphs, so Russian translations would render as empty boxes. Noto Sans is
# bundled in the repo (SIL OFL, see fonts/OFL.txt) so this works with zero
# system dependencies, regardless of OS/fonts installed.
_FONT_FACE_CSS = f"""
<style>
  @font-face {{ font-family: "NotoSansFrank"; src: url({_FONTS_DIR / "NotoSans-Regular.ttf"}); }}
  @font-face {{ font-family: "NotoSansFrank"; font-weight: bold; src: url({_FONTS_DIR / "NotoSans-Bold.ttf"}); }}
  @font-face {{ font-family: "NotoSansFrank"; font-style: italic; src: url({_FONTS_DIR / "NotoSans-Italic.ttf"}); }}
  body, * {{ font-family: "NotoSansFrank", sans-serif !important; }}
</style>
"""


class PdfNotAvailable(Exception):
    pass


def render_pdf(html: str) -> bytes:
    from xhtml2pdf import pisa

    patched_html = html.replace("</head>", _FONT_FACE_CSS + "</head>")
    out = BytesIO()
    result = pisa.CreatePDF(patched_html, dest=out)
    if result.err:
        raise PdfNotAvailable(f"xhtml2pdf reported errors while rendering the PDF: {result.err}")
    return out.getvalue()
