"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value

@dataclass
class ReflectionConfig:
    """
    Reflection Layer config with backward-compat.
    - 支持两套字段/ENV 名（旧：ENABLE_REFLECTION/REFLECTION_TOP_K/...；新：REFLECTION_ENABLED/...）
    - 暴露统一的新字段：enabled / max_iters / min_support / min_coverage / max_contradiction / min_attributable / targeted_topk
    - 同时保留旧字段：enable_reflection / reflection_top_k / reflection_* 以兼容旧代码
    """

    # ===== 旧字段（兼容旧脚本）=====
    enable_reflection: bool = field(
        default=get_env_value("ENABLE_REFLECTION",
                 get_env_value("REFLECTION_ENABLED", True, bool), bool)
    )
    reflection_top_k: int = field(
        default=get_env_value("REFLECTION_TOP_K",
                 get_env_value("REFLECTION_TOPK", 6, int), int)
    )
    reflection_query_mode: str = field(
        default=get_env_value("REFLECTION_QUERY_MODE", "hybrid", str)
    )
    reflection_temperature: float = field(
        default=get_env_value("REFLECTION_TEMPERATURE", 0.1, float)
    )
    reflection_max_sentences: int = field(
        default=get_env_value("REFLECTION_MAX_SENTENCES", 20, int)
    )
    reflection_support_threshold: float = field(
        default=get_env_value("REFLECTION_SUPPORT_THRESHOLD",
                 get_env_value("REFLECTION_MIN_SUPPORT", 0.70, float), float)
    )
    reflection_enable_contradiction_check: bool = field(
        default=get_env_value("REFLECTION_ENABLE_CONTRADICTION_CHECK", True, bool)
    )
    reflection_enable_coverage_check: bool = field(
        default=get_env_value("REFLECTION_ENABLE_COVERAGE_CHECK", True, bool)
    )
    reflection_enable_attribution: bool = field(
        default=get_env_value("REFLECTION_ENABLE_ATTRIBUTION", True, bool)
    )

    # ===== 新字段（推荐上层代码使用）=====
    enabled: bool = field(init=False)
    max_iters: int = field(default=get_env_value("REFLECTION_MAX_ITERS", 2, int))
    min_support: float = field(
        default=get_env_value("REFLECTION_MIN_SUPPORT",
                 get_env_value("REFLECTION_SUPPORT_THRESHOLD", 0.70, float), float)
    )
    min_coverage: float = field(default=get_env_value("REFLECTION_MIN_COVERAGE", 0.85, float))
    max_contradiction: float = field(default=get_env_value("REFLECTION_MAX_CONTRADICTION", 0.10, float))
    min_attributable: float = field(default=get_env_value("REFLECTION_MIN_ATTRIBUTABLE", 0.80, float))
    targeted_topk: int = field(init=False)

    def __post_init__(self):
        # 新字段由旧字段桥接，双向兼容
        self.enabled = bool(self.enable_reflection)
        self.targeted_topk = int(self.reflection_top_k)
        # 保底同步旧字段
        self.reflection_support_threshold = float(self.min_support)
        self.reflection_top_k = int(self.targeted_topk)
        self.enable_reflection = bool(self.enabled)

@dataclass
class RAGAnythingConfig:
    """Configuration class for RAGAnything with environment variable support"""

    # Directory Configuration
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./rag_storage", str))
    """Directory where RAG storage and cache files are stored."""

    # Parser Configuration
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """Default parsing method for document parsing: 'auto', 'ocr', or 'txt'."""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """Default output directory for parsed content."""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """Parser selection: 'mineru' or 'docling'."""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """Whether to display content statistics during parsing."""

    # Multimodal Processing Configuration
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """Enable image content processing."""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """Enable table content processing."""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """Enable equation content processing."""

    # Batch Processing Configuration
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """Maximum number of files to process concurrently."""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
            str,
        ).split(",")
    )
    """List of supported file extensions for batch processing."""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """Whether to recursively process subfolders in batch mode."""

    # Context Extraction Configuration
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """Number of pages/chunks to include before and after current item for context."""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )
    """Maximum number of tokens in extracted context."""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """Whether to include document headers and titles in context."""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """Whether to include image/table captions in context."""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", "text", str
        ).split(",")
    )
    """Content types to include in context extraction (e.g., 'text', 'image', 'table')."""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """Default content format for context extraction when processing documents."""

    # Micro Planner Configuration
    # ---
    enable_micro_planner: bool = field(
        default=get_env_value("ENABLE_MICRO_PLANNER", False, bool)
    )
    """Enable the query micro planner."""

    query_time_budget_ms: int = field(
        default=get_env_value("QUERY_TIME_BUDGET_MS", 1000, int)
    )
    """Approximate time budget per query in milliseconds."""

    memory_budget_gb: float = field(
        default=get_env_value("MEMORY_BUDGET_GB", 2.0, float)
    )
    """Approximate memory budget in gigabytes for planning policies."""

    default_chunk_top_k: int = field(
        default=get_env_value("DEFAULT_CHUNK_TOP_K", 5, int)
    )
    """Default chunk_top_k used when planner does not specify one."""

    enable_reflection: bool = field(
        default=get_env_value("ENABLE_REFLECTION", False, bool)
    )
    """Enable answer reflection step after generation."""

    def __post_init__(self):
        """Post-initialization setup for backward compatibility"""
        # Support legacy environment variable names for backward compatibility
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def mineru_parse_method(self) -> str:
        """
        Backward compatibility property for old code.

        .. deprecated::
           Use `parse_method` instead. This property will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """Setter for backward compatibility"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
