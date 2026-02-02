from typing import ClassVar, Dict, Optional, Union
import os
import httpx

# ═══════════════════════════════════════════════════════════════
# ⭐ NEUER ANSATZ: ChatOpenAI direkt patchen
# ═══════════════════════════════════════════════════════════════

from loguru import logger

# Globaler persistenter HTTP Client
_GLOBAL_HTTP_CLIENT = None

def get_persistent_http_client():
    """Gibt einen persistenten HTTP Client zurück, der NIEMALS geschlossen wird"""
    global _GLOBAL_HTTP_CLIENT
    
    if _GLOBAL_HTTP_CLIENT is None:
        timeout_value = float(os.getenv("OPENAI_TIMEOUT", "120"))
        
        _GLOBAL_HTTP_CLIENT = httpx.Client(
            timeout=httpx.Timeout(
                timeout=timeout_value,
                connect=60.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=300.0
            ),
        )
        logger.info(f"🔧 Globaler HTTP Client erstellt mit timeout={timeout_value}s")
    
    return _GLOBAL_HTTP_CLIENT

# Patch ChatOpenAI statt OpenAI direkt
try:
    from langchain_openai import ChatOpenAI
    
    _original_chatopenai_init = ChatOpenAI.__init__
    
    def patched_chatopenai_init(self, *args, **kwargs):
        """Patched ChatOpenAI.__init__ um persistenten Client zu erzwingen"""
        # Setze http_client wenn nicht vorhanden
        if 'http_client' not in kwargs:
            kwargs['http_client'] = get_persistent_http_client()
            logger.debug("🔧 Persistenter HTTP Client in ChatOpenAI injiziert")
        
        # Setze auch http_async_client auf None um zu verhindern dass ein neuer erstellt wird
        if 'http_async_client' not in kwargs:
            kwargs['http_async_client'] = None
        
        return _original_chatopenai_init(self, *args, **kwargs)
    
    ChatOpenAI.__init__ = patched_chatopenai_init
    logger.success("✅ ChatOpenAI HTTP Client Patch angewendet")
    
except ImportError as e:
    logger.warning(f"⚠️ ChatOpenAI nicht verfügbar: {e}")
except Exception as e:
    logger.error(f"❌ Fehler beim Patchen von ChatOpenAI: {e}")

# Zusätzlich: OpenAI base client auch patchen (doppelte Sicherheit)
try:
    from openai import OpenAI
    
    _original_openai_init = OpenAI.__init__
    
    def patched_openai_init(self, *args, **kwargs):
        if 'http_client' not in kwargs:
            kwargs['http_client'] = get_persistent_http_client()
            logger.debug("🔧 Persistenter HTTP Client in OpenAI injiziert")
        return _original_openai_init(self, *args, **kwargs)
    
    OpenAI.__init__ = patched_openai_init
    logger.success("✅ OpenAI HTTP Client Patch angewendet")
    
except ImportError:
    logger.warning("⚠️ OpenAI nicht installiert")
except Exception as e:
    logger.error(f"❌ Fehler beim Patchen von OpenAI: {e}")

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
        # KEIN lokaler _http_client mehr - wir nutzen den globalen
        logger.info("ModelManager initialisiert")

    async def get_model(self, model_id: str, **kwargs) -> Optional[ModelType]:
        """
        Holt ein Model nach ID.
        Für OpenAI: Nutzt ChatOpenAI direkt (Esperanto wird umgangen).
        """
        if not model_id:
            return None

        try:
            model: Model = await Model.get(model_id)
        except Exception:
            raise ValueError(f"Model mit ID {model_id} nicht gefunden")

        if not model.type or model.type not in [
            "language",
            "embedding",
            "speech_to_text",
            "text_to_speech",
        ]:
            raise ValueError(f"Ungültiger Model-Typ: {model.type}")

        # ⭐ FIX: Für OpenAI language models, ChatOpenAI direkt nutzen
        if model.type == "language" and model.provider.lower() == "openai":
            from langchain_openai import ChatOpenAI
            
            logger.debug(f"Erstelle ChatOpenAI direkt: {model.name}")
            
            # Der Patch oben sorgt dafür dass http_client automatisch gesetzt wird
            return ChatOpenAI(
                model=model.name,
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=float(os.getenv("OPENAI_TIMEOUT", "120")),
                max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                streaming=kwargs.get("streaming", False),
                # http_client wird automatisch durch Patch gesetzt
                **{k: v for k, v in kwargs.items() if k not in ['streaming', 'timeout', 'max_retries']}
            )
        
        # Für andere Provider: Esperanto
        elif model.type == "language":
            kwargs["streaming"] = False
            kwargs["timeout"] = 120
            kwargs["max_retries"] = 3
            
            logger.debug(f"Erstelle Language Model via Esperanto: {model.name} ({model.provider})")
            
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
            raise ValueError(f"Ungültiger Model-Typ: {model.type}")

    async def get_defaults(self) -> DefaultModels:
        """Holt die Default Models Konfiguration aus der Datenbank"""
        defaults = await DefaultModels.get_instance()
        if not defaults:
            raise RuntimeError("Fehler beim Laden der Default Models Konfiguration")
        return defaults

    async def get_speech_to_text(self, **kwargs) -> Optional[SpeechToTextModel]:
        """Holt das Default Speech-to-Text Model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_speech_to_text_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, SpeechToTextModel), (
            f"Erwartete SpeechToTextModel aber bekam {type(model)}"
        )
        return model

    async def get_text_to_speech(self, **kwargs) -> Optional[TextToSpeechModel]:
        """Holt das Default Text-to-Speech Model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_text_to_speech_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, TextToSpeechModel), (
            f"Erwartete TextToSpeechModel aber bekam {type(model)}"
        )
        return model

    async def get_embedding_model(self, **kwargs) -> Optional[EmbeddingModel]:
        """Holt das Default Embedding Model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_embedding_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, EmbeddingModel), (
            f"Erwartete EmbeddingModel aber bekam {type(model)}"
        )
        return model

    async def get_default_model(self, model_type: str, **kwargs) -> Optional[ModelType]:
        """Holt das Default Model für einen bestimmten Typ"""
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
                f"Kein Default Model konfiguriert für Typ '{model_type}'. "
                f"Bitte gehe zu Einstellungen → Models und setze ein Default Model."
            )
            return None

        try:
            return await self.get_model(model_id, **kwargs)
        except ValueError as e:
            logger.error(
                f"Fehler beim Laden des Default Models für Typ '{model_type}': {e}. "
                f"Die konfigurierte model_id '{model_id}' wurde möglicherweise gelöscht oder ist falsch konfiguriert. "
                f"Bitte gehe zu Einstellungen → Models und konfiguriere das Default Model neu."
            )
            return None


model_manager = ModelManager()
