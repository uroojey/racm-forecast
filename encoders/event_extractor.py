import os
import json
import re
import logging
from typing import Optional, List, Dict

from .event_schema import FinancialEvent, EventBatch
from .prompt_templates import PromptTemplates
from .cache import EventCache
from .ner_postprocessor import NERPostprocessor

# Module-level import so it can be cleanly mocked in tests
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

class EventExtractor:
    """
    Extracts structured FinancialEvents from raw news text.
    Supports two backends:
    - "openai": Uses OpenAI GPT API (gpt-3.5-turbo or gpt-4)
    - "local": Uses a local HuggingFace model (e.g. Llama)
    """
    def __init__(
        self,
        backend: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        cache_dir: str = ".cache/events",
        ticker_map: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ):
        self.backend = backend
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache = EventCache(cache_dir=cache_dir)
        self.ner = NERPostprocessor(ticker_map=ticker_map)

        if self.backend == "openai":
            if OpenAI is None:
                raise ImportError("openai package is required for backend='openai'. Run `pip install openai>=1.0.0`")
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var.")
            self.client = OpenAI(api_key=key)
        elif self.backend == "local":
            self.local_pipeline = None  # Lazy load
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _call_openai(self, text: str) -> str:
        """Call OpenAI API with the extraction prompt."""
        prompt = PromptTemplates.extraction_prompt(text)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content or "[]"

    def _call_local(self, text: str) -> str:
        """Call local HuggingFace pipeline (text-generation)."""
        if self.local_pipeline is None:
            try:
                import transformers
                logger.info(f"Loading local model: {self.model_name}")
                self.local_pipeline = transformers.pipeline("text-generation", model=self.model_name)
            except ImportError:
                raise ImportError("transformers package required for backend='local'.")
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")
        
        prompt = f"{PromptTemplates.SYSTEM_PROMPT}\n\n{PromptTemplates.extraction_prompt(text)}"
        result = self.local_pipeline(prompt, max_new_tokens=512, temperature=0.1, return_full_text=False)
        return result[0]['generated_text']

    def _parse_response(self, raw: str, source_text: str) -> List[FinancialEvent]:
        """Parse the LLM JSON string into a list of FinancialEvent objects."""
        cleaned = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                logger.warning("LLM output is not a JSON list. Wrapping in list.")
                data = [data]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}\nRaw output: {raw}")
            return []

        events = []
        for item in data:
            try:
                item["source_text"] = source_text
                if "ticker" not in item:
                    item["ticker"] = None
                
                event = FinancialEvent.from_dict(item)
                if event.validate():
                    events.append(event)
                else:
                    logger.warning(f"Event failed validation: {item}")
            except Exception as e:
                logger.warning(f"Error instantiating FinancialEvent: {e}")
        
        return events

    def extract(self, text: str, date: Optional[str] = None) -> EventBatch:
        """Extract events from a single text."""
        raw_response = None
        if self.use_cache:
            raw_response = self.cache.get(text)
        
        if not raw_response:
            try:
                if self.backend == "openai":
                    raw_response = self._call_openai(text)
                else:
                    raw_response = self._call_local(text)
                
                if self.use_cache and raw_response:
                    self.cache.set(text, raw_response)
            except Exception as e:
                logger.error(f"Error during extraction: {e}")
                return EventBatch(events=[], source_date=date)
        
        events = self._parse_response(raw_response, text)
        events = self.ner.enrich_events(events, text)
        
        return EventBatch(events=events, source_date=date)

    def extract_batch(self, texts: List[str], dates: Optional[List[str]] = None) -> List[EventBatch]:
        """Extract events from a list of texts."""
        if dates is None:
            dates = [None] * len(texts)
        return [self.extract(text, date) for text, date in zip(texts, dates)]
