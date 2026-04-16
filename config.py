import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL: str = "claude-sonnet-4-6"
    MAX_ITERATIONS: int = 20
    SANDBOX_TIMEOUT: int = 10
    VALIDATION_TIMEOUT: int = 15
    TOOL_GEN_RETRIES: int = 3
    LIBRARY_PATH: str = os.path.join(os.path.dirname(__file__), "library", "tool_library.db")
    PORT: int = int(os.getenv("PORT", "5001"))
    FAILURES_PATH: str = os.path.join(os.path.dirname(__file__), "library", "failures.db")
    # Retrieval
    RETRIEVAL_THRESHOLD: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.75"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    # Ablation flags (set via env vars for eval runs)
    ABLATION_NO_LIBRARY: bool = os.getenv("ABLATION_NO_LIBRARY", "false").lower() == "true"
    ABLATION_NO_ABSTRACTION: bool = os.getenv("ABLATION_NO_ABSTRACTION", "false").lower() == "true"
    ABLATION_NO_LIBRARIAN: bool = os.getenv("ABLATION_NO_LIBRARIAN", "false").lower() == "true"
    # Workspace directory for file operations
    WORKSPACE_DIR: str = os.getenv("WORKSPACE_DIR", os.getcwd())
