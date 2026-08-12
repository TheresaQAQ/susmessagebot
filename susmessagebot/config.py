import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH")

# SiliconFlow (OpenAI-compatible). OPENROUTER_* kept as fallback aliases.
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENROUTER_API_KEY")
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    os.getenv("OPENROUTER_BASE_URL", "https://api.siliconflow.cn/v1"),
)
# Default matches README / SiliconFlow free-tier docs; override via SILICONFLOW_MODEL.
SILICONFLOW_MODEL = os.getenv(
    "SILICONFLOW_MODEL",
    os.getenv("OPENROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
)
# Alibaba Cloud Model Studio (DashScope) OpenAI-compatible fallback backend.
# Image moderation always uses it; text and URL moderation use it as fallback.
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_VISION_MODEL = os.getenv(
    "DASHSCOPE_VISION_MODEL",
    "qwen3-vl-flash",
)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
# Discord user ID shown in ban DMs for unban appeals (add this user to send feedback).
APPEAL_DISCORD_USER_ID = os.getenv("APPEAL_DISCORD_USER_ID", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 1.0
MAX_EXAMPLES = 5

# Runtime data directory. Local and container runs both keep state under data/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(
    os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
)
os.makedirs(DATA_DIR, exist_ok=True)
STATS_DB_PATH = os.path.join(DATA_DIR, "stats.db")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8001"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
