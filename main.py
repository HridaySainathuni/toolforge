import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from library.tool_library import ToolLibrary
from web.app import init_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

log = logging.getLogger(__name__)


def main() -> None:
    if not Config.ANTHROPIC_API_KEY or Config.ANTHROPIC_API_KEY == "your-api-key-here":
        log.error("Set ANTHROPIC_API_KEY in toolforge/.env")
        sys.exit(1)

    log.info("Loading tool library from %s", Config.LIBRARY_PATH)
    library = ToolLibrary(Config.LIBRARY_PATH)
    log.info("Loaded %d tools", len(library.tools))

    app = init_app(library)

    log.info("Starting ToolForge on http://localhost:%d", Config.PORT)
    app.run(host="0.0.0.0", port=Config.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
