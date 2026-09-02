from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into the actual process environment. Pydantic's own env_file
# support below only populates fields DECLARED on this Settings class — it
# does not set them as real os.environ values. GROQ_API_KEY is deliberately
# NOT a Settings field (see app/rag/generation.py — it's read via plain
# os.getenv so the Groq integration stays fully optional), so without this
# explicit load, a .env file containing GROQ_API_KEY would silently never
# be seen by that lookup.
load_dotenv()


class Settings(BaseSettings):
    """Central configuration, overridable via environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    app_name: str = "Arabic-Aware Document Intelligence for RAG"

    # SQL store
    database_url: str = "sqlite:///./data/app.db"

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"

    # Embeddings — multilingual model so English and Arabic share one vector space
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Chunking
    fixed_chunk_size: int = 800          # characters
    fixed_chunk_overlap: int = 120       # characters
    dynamic_chunk_max_size: int = 1500   # characters, soft cap per structural chunk
    min_chunk_size: int = 200            # avoid tiny orphan chunks

    # Arabic handling
    strip_diacritics_default: bool = False  # preserve tashkeel unless caller opts out

    # Retrieval
    default_top_k: int = 5


settings = Settings()
