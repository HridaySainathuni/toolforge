"""One-time script to re-embed seed tools in the library with real embeddings."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from library.tool_library import ToolLibrary
from agent.retriever import embed


def main() -> None:
    lib = ToolLibrary(db_path=Config.LIBRARY_PATH, seed=True)
    tools = lib.get_all()
    if not tools:
        print("No tools in library.")
        return
    for tool in tools:
        desc = tool["description"]
        emb = embed(desc)
        lib.replace_tool(tool["name"], new_source=tool["source_code"], new_embedding=emb)
        print(f"Re-embedded: {tool['name']}")
    print(f"Done. Re-embedded {len(tools)} tools.")


if __name__ == "__main__":
    main()
