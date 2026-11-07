"""Central configuration, loaded from environment / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_prefix="APP_", extra="ignore"
    )

    # LLM
    llm_provider: str = "none"           # none | openai | gemini
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None

    # Runtime
    device: str = "auto"                 # auto | cpu | cuda
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    runs_dir: Path = PROJECT_ROOT / "runs"
    max_upload_mb: int = 10

    # Vision (PaDiM)
    vision_backbone: str = "resnet18"    # timm feature extractor
    vision_input_size: int = 256         # center-crop size
    padim_n_features: int = 100          # random dims kept (paper: 100 for R18)
    anomaly_threshold: float = 0.5       # image-level normalised score threshold
    max_defect_regions: int = 8          # connected components returned

    # Retrieval
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retriever_top_k: int = 5
    retriever_mode: str = "hybrid"       # dense | bm25 | hybrid

    # Agent
    agent_max_iters: int = 8
    agent_deterministic: bool = False    # force rule-based tool selection

    @property
    def allowed_image_ext(self) -> set[str]:
        return {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: F401
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
