from abc import ABC
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq
import logging
from typing import Dict, Type, List
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def getModelNames() -> List[str]:
    """
    Fetch available models from local Ollama instance.
    Returns model names with " (local)" suffix for display.
    Falls back to default list if Ollama is not available.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if models:
                return [f"{model['name']} (local)" for model in models]
    except Exception as e:
        logging.warning(f"Failed to fetch Ollama models: {e}")
    
    # Fallback to default models if Ollama is not available
    return [
        "llama3.2 (local)",
    ]


class LLMException(Exception):
    pass


class LLM(ABC):
    """
    An abstract superclass for interacting with LLMs - subclass for Claude and GPT
    """

    model_names = []

    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.client = None
        self.temperature = temperature
        self.reasoning_effort = None

    def send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        result = self.protected_send(system, user, max_tokens)
        left = result.find("{")
        right = result.rfind("}")
        if left > -1 and right > -1:
            result = result[left: right + 1]
        return result

    def protected_send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        retries = 3
        while retries:
            retries -= 1
            try:
                return self._send(system, user, max_tokens)
            except Exception as e:
                logging.error(f"Exception on calling LLM of {e}")
                if retries:
                    logging.warning("Waiting 2s and retrying")
                    time.sleep(2)
        return "{}"

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        if self.reasoning_effort:
            response = self.client.chat.completions.create(
                model=self.api_model_name(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=self.reasoning_effort,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.api_model_name(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        return response.choices[0].message.content

    def api_model_name(self) -> str:
        """
        Return the actual model name to pass to the API.
        For Ollama models displayed as 'llama3.2 (local)', strip the suffix.
        For other models, strip anything after a space as before.
        """
        name = self.model_name
        # Strip ' (local)' suffix added for display — remainder is the exact Ollama model name
        if name.endswith(" (local)"):
            return name[: -len(" (local)")]
        # Original behaviour: strip anything after first space
        if " " in name:
            return name.split(" ")[0]
        return name

    @classmethod
    def model_map(cls) -> Dict[str, Type[Self]]:
        mapping = {}
        for llm in cls.__subclasses__():
            for model_name in llm.model_names:
                mapping[model_name] = llm
        return mapping

    @classmethod
    def all_supported_model_names(cls) -> List[str]:
        return list(cls.model_map().keys())

    @classmethod
    def all_model_names(cls) -> List[str]:
        """
        Return only models that have their API key configured.
        Filters out unavailable models to keep UI clean.
        """
        # Map each LLM class to its required environment variable
        api_key_requirements = {
            'GPT': 'OPENAI_API_KEY',
            'Claude': 'ANTHROPIC_API_KEY',
            'GroqAPI': 'GROQ_API_KEY',
            'Gemini': 'GOOGLE_API_KEY',
            'DeepSeekAPI': 'DEEPSEEK_API_KEY',
            'Ollama': None,  # Ollama checks availability itself
        }
        
        available_models = []
        for llm_subclass in cls.__subclasses__():
            class_name = llm_subclass.__name__
            
            # Special handling for Ollama (checks localhost availability)
            if class_name == 'Ollama':
                if llm_subclass._is_ollama_available():
                    available_models.extend(llm_subclass.get_model_names())
            else:
                # For other providers, check if API key is set
                required_key = api_key_requirements.get(class_name)
                if required_key and os.getenv(required_key):
                    available_models.extend(llm_subclass.model_names)
        
        # Allow manual override via MODELS env var
        allowed = os.getenv("MODELS")
        if allowed:
            allowed_models = allowed.split(",")
            return [model for model in allowed_models if model in available_models]
        
        return available_models

    @classmethod
    def create(cls, model_name: str, temperature: float = 0.5) -> Self:
        subclass = cls.model_map().get(model_name)
        if not subclass:
            raise LLMException(f"Unrecognized LLM model name specified: {model_name}")
        return subclass(model_name, temperature)


class Claude(LLM):
    model_names = [
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ]

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = Anthropic()

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        response = self.client.messages.create(
            model=self.api_model_name(),
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class GPT(LLM):
    model_names = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = OpenAI()
        if "gpt-5" in model_name:
            self.reasoning_effort = "low"


class O1(LLM):
    model_names = []

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = OpenAI()

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        message = system + "\n\n" + user
        response = self.client.chat.completions.create(
            model=self.api_model_name(),
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content


class O3(LLM):
    model_names = []

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        override = os.getenv("OPENAI_API_KEY_O3")
        if override:
            print("Using special key with o3 access")
            self.client = OpenAI(api_key=override)
        else:
            self.client = OpenAI()

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        message = system + "\n\n" + user
        response = self.client.chat.completions.create(
            model=self.api_model_name(),
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content


class Gemini(LLM):
    model_names = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )


class Ollama(LLM):
    """
    Interface to local Ollama models via the OpenAI-compatible API.
    Only available when running locally with Ollama accessible on localhost:11434.
    """
    model_names = []  # Lazy-loaded, empty on HF Spaces
    _ollama_available = None  # Cache availability check
    
    @classmethod
    def _is_ollama_available(cls) -> bool:
        """Check if Ollama is actually running locally"""
        if cls._ollama_available is not None:
            return cls._ollama_available
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            cls._ollama_available = response.status_code == 200
        except:
            cls._ollama_available = False
        return cls._ollama_available

    @classmethod
    def get_model_names(cls):
        """Only return Ollama models if Ollama is actually running"""
        if not cls.model_names and cls._is_ollama_available():
            cls.model_names = getModelNames()
        return cls.model_names

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        response = self.client.chat.completions.create(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        reply = response.choices[0].message.content
        if "</think>" in reply:
            logging.info("Thoughts:\n" + reply.split("</think>")[0].replace("<think>", ""))
            reply = reply.split("</think>")[1]
        return reply


class DeepSeekAPI(LLM):
    model_names = ["deepseek-chat V3", "deepseek-reasoner R1"]

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )


class DeepSeekLocal(LLM):
    model_names = []

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        system += "\nImportant: avoid overthinking. Think briefly and decisively. The final response must follow the given json format or you forfeit the game. Do not overthink. Respond with json."
        user += "\nImportant: avoid overthinking. Think briefly and decisively. The final response must follow the given json format or you forfeit the game. Do not overthink. Respond with json."
        response = self.client.chat.completions.create(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        reply = response.choices[0].message.content
        if "</think>" in reply:
            logging.info("Thoughts:\n" + reply.split("</think>")[0].replace("<think>", ""))
            reply = reply.split("</think>")[1]
        return reply


class GroqAPI(LLM):
    model_names = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        self.client = Groq()
