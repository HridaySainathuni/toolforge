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
    LIBRARY_PATH: str = os.path.join(os.path.dirname(__file__), "library", "tool_library.json")
    PORT: int = 5000
