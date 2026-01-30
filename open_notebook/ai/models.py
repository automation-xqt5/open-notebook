from typing import ClassVar, Dict, Optional, Union
import os
import httpx

# ═══════════════════════════════════════════════════════════════
# ⭐ OPENAI HTTP CLIENT PATCH - Direkt hier integriert
# ═══════════════════════════════════════════════════════════════

from loguru import logger

# Globaler persistenter HTTP Client
_GLOBAL_HTTP_CLIENT = None

def get_persistent_http_client():
    """Gibt einen persistenten HTTP Client zurück"""
    global _GLOBAL_HTTP_CLIENT
    
    if _GLOBAL_HTTP_CLIENT is None:
        timeout_value = float(os.getenv("OPENAI_TIMEOUT", "120"))
        
        _GLOBAL_HTTP_CLIENT = httpx.Client(
            timeout=httpx.Timeout(
                timeout=timeout_value,
                connect=60.0,
                read=timeout_value,
                write=30.0,
                pool=5.0
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=300.0
            ),
            http1=True,
            http2=False,
        )
        logger.info(f"🔧 Persistent HTTP client created with timeout={timeout_value}s")
    
    return _GLOBAL_HTTP_CLIENT

# Monkey Patch für OpenAI
try:
    from openai import OpenAI
    
    _original_openai_init = OpenAI.__init__
    
    def patched_openai_init(self, *args, **kwargs):
        if 'http_client' not in kwargs:
            kwargs['http_client'] = get_persistent_http_client()
            logger.debug("🔧 Injected persistent HTTP client into OpenAI")
        return _original_openai_init(self, *args, **kwargs)
    
    OpenAI.__init__ = patched_openai_init
    logger.success("✅ OpenAI HTTP client patch applied")
    
except ImportError:
    logger.warning("⚠️ OpenAI not installed, skipping patch")
except Exception as e:
    logger.error(f"❌ Failed to apply OpenAI patch: {e}")

# ═══════════════════════════════════════════════════════════════
# REST DES ORIGINALEN CODES
# ═══════════════════════════════════════════════════════════════

from esperanto import (
    AIFactory,
    EmbeddingModel,
    LanguageModel,
    SpeechToTextModel,
    TextToSpeechModel,
)

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.base import ObjectModel, RecordModel

ModelType = Union[LanguageModel, EmbeddingModel, SpeechToTextModel, TextToSpeechModel]


class Model(ObjectModel):
    table_name: ClassVar[str] = "model"
    name: str
    provider: str
    type: str

    @classmethod
    async def get_models_by_type(cls, model_type):
        models = await repo_query(
            "SELECT * FROM model WHERE type=$model_type;", {"model_type": model_type}
        )
        return [Model(**model) for model in models]


class DefaultModels(RecordModel):
    record_id: ClassVar[str] = "open_notebook:default_models"
    default_chat_model: Optional[str] = None
    default_transformation_model: Optional[str] = None
    large_context_model: Optional[str] = None
    default_text_to_speech_model: Optional[str] = None
    default_speech_to_text_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_tools_model: Optional[str] = None

    @classmethod
    async def get_instance(cls) -> "DefaultModels":
        """Always fetch fresh defaults from database"""
        result = await repo_query(
            "SELECT * FROM ONLY $record_id",
            {"record_id": ensure_record_id(cls.record_id)},
        )

        if result:
            if isinstance(result, list) and len(result) > 0:
                data = result[0]
            elif isinstance(result, dict):
                data = result
            else:
                data = {}
        else:
            data = {}

        instance = object.__new__(cls)
        object.__setattr__(instance, "__dict__", {})
        super(RecordModel, instance).__init__(**data)
        return instance


class ModelManager:
    def __init__(self):
        self._http_client = None
        logger.info("ModelManager initialized")

    def _get_http_client(self):
        """Lazy initialization des HTTP Clients"""
        if self._http_client is None:
            self._http_client = get_persistent_http_client()
        return self._http_client

    async def get_model(self, model_id: str, **kwargs) -> Optional[ModelType]:
        """Get a model by ID. For OpenAI, use ChatOpenAI directly."""
        if not model_id:
            return None

        try:
            model: Model = await Model.get(model_id)
        except Exception:
            raise ValueError(f"Model with ID {model_id} not found")

        if not model.type or model.type not in [
            "language",
            "embedding",
            "speech_to_text",
            "text_to_speech",
        ]:
            raise ValueError(f"Invalid model type: {model.type}")

        # ⭐ FIX: Für OpenAI language models, ChatOpenAI direkt nutzen
        if model.type == "language" and model.provider.lower() == "openai":
            from langchain_openai import ChatOpenAI
            
            logger.debug(f"Creating ChatOpenAI directly: {model.name}")
            
            return ChatOpenAI(
                model=model.name,
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=float(os.getenv("OPENAI_TIMEOUT", "120")),
                max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                streaming=kwargs.get("streaming", False),
                http_client=self._get_http_client(),
                **{k: v for k, v in kwargs.items() if k not in ['streaming', 'timeout', 'max_retries']}
            )
        
        # Für andere Provider: Esperanto
        elif model.type == "language":
            kwargs["streaming"] = False
            kwargs["timeout"] = 120
            kwargs["max_retries"] = 3
            
            logger.debug(f"Creating language model via Esperanto: {model.name} ({model.provider})")
            
            return AIFactory.create_language(
                model_name=model.name,
                provider=model.provider,
                config=kwargs,
            )
        elif model.type == "embedding":
            return AIFactory.create_embedding(
                model_name=model.name,
                provider=model.provider,
                config=kwargs,
            )
        elif model.type == "speech_to_text":
            return AIFactory.create_speech_to_text(
                model_name=model.name,
                provider=model.provider,
                config=kwargs,
            )
        elif model.type == "text_to_speech":
            return AIFactory.create_text_to_speech(
                model_name=model.name,
                provider=model.provider,
                config=kwargs,
            )
        else:
            raise ValueError(f"Invalid model type: {model.type}")

    async def get_defaults(self) -> DefaultModels:
        """Get the default models configuration from database"""
        defaults = await DefaultModels.get_instance()
        if not defaults:
            raise RuntimeError("Failed to load default models configuration")
        return defaults

    async def get_speech_to_text(self, **kwargs) -> Optional[SpeechToTextModel]:
        """Get the default speech-to-text model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_speech_to_text_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, SpeechToTextModel), (
            f"Expected SpeechToTextModel but got {type(model)}"
        )
        return model

    async def get_text_to_speech(self, **kwargs) -> Optional[TextToSpeechModel]:
        """Get the default text-to-speech model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_text_to_speech_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, TextToSpeechModel), (
            f"Expected TextToSpeechModel but got {type(model)}"
        )
        return model

    async def get_embedding_model(self, **kwargs) -> Optional[EmbeddingModel]:
        """Get the default embedding model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_embedding_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, EmbeddingModel), (
            f"Expected EmbeddingModel but got {type(model)}"
        )
        return model

    async def get_default_model(self, model_type: str, **kwargs) -> Optional[ModelType]:
        """Get the default model for a specific type."""
        defaults = await self.get_defaults()
        model_id = None

        if model_type == "chat":
            model_id = defaults.default_chat_model
        elif model_type == "transformation":
            model_id = (
                defaults.default_transformation_model or defaults.default_chat_model
            )
        elif model_type == "tools":
            model_id = defaults.default_tools_model or defaults.default_chat_model
        elif model_type == "embedding":
            model_id = defaults.default_embedding_model
        elif model_type == "text_to_speech":
            model_id = defaults.default_text_to_speech_model
        elif model_type == "speech_to_text":
            model_id = defaults.default_speech_to_text_model
        elif model_type == "large_context":
            model_id = defaults.large_context_model

        if not model_id:
            logger.warning(
                f"No default model configured for type '{model_type}'. "
                f"Please go to Settings → Models and set a default model."
            )
            return None

        try:
            return await self.get_model(model_id, **kwargs)
        except ValueError as e:
            logger.error(
                f"Failed to load default model for type '{model_type}': {e}. "
                f"The configured model_id '{model_id}' may have been deleted or misconfigured. "
                f"Please go to Settings → Models and reconfigure the default model."
            )
            return None

    def __del__(self):
        """Cleanup HTTP client on destroy"""
        try:
            if hasattr(self, '_http_client') and self._http_client is not None:
                self._http_client.close()
                logger.debug("HTTP client closed")
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")


model_manager = ModelManager()
