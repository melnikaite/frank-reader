"""Console entry point: `frank-reader` starts the web server.

Host/port come from Settings (FRANK_HOST / FRANK_PORT env vars or .env).
"""

import uvicorn

from frank_reader.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run("frank_reader.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
