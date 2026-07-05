import io
from pathlib import Path

from PIL import Image

from frank_reader.adapters.base import PageContent
from frank_reader.config import Settings


class ImageAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self, source: Path) -> list[PageContent]:
        data = Path(source).read_bytes()
        im = Image.open(io.BytesIO(data))
        im.load()
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        max_dim = max(im.width, im.height)
        if max_dim > self.settings.page_render_max_dim:
            scale = self.settings.page_render_max_dim / max_dim
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return [PageContent(page_number=1, kind="image", image_png=buf.getvalue())]
