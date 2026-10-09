"""Unit tests for Module 3 — LLM-based Event Extraction.

These tests do NOT require an OpenAI API key. They mock the LLM calls
and test parsing, schema validation, caching, and NER enrichment.
"""
import json
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_NEWS = (
    "Apple Inc. reported a 20% decline in quarterly profit on Wednesday, "
    "missing analyst expectations. CEO Tim Cook cited supply chain issues."
)

SAMPLE_EVENTS_JSON = json.dumps([
    {
        "company": "Apple Inc.",
        "event_type": "earnings",
        "description": "Apple reported 20% decline in quarterly profit.",
        "impact": "negative",
        "impact_magnitude": 0.7,
        "date": None,
        "confidence": 0.92
    }
])

SAMPLE_EVENTS_FENCED = f"```json\n{SAMPLE_EVENTS_JSON}\n```"


# ── FinancialEvent schema tests ───────────────────────────────────────────────

class TestFinancialEvent:

    def test_valid_event(self):
        from encoders.event_schema import FinancialEvent
        event = FinancialEvent(
            company="Apple Inc.",
            ticker="AAPL",
            event_type="earnings",
            description="Apple missed earnings estimates.",
            impact="negative",
            impact_magnitude=0.7,
            date="2024-01-15",
            source_text=SAMPLE_NEWS,
            confidence=0.9
        )
        assert event.validate() is True

    def test_invalid_event_type(self):
        from encoders.event_schema import FinancialEvent
        event = FinancialEvent(
            company="Apple", ticker=None, event_type="INVALID_TYPE",
            description="Some event.", impact="negative",
            impact_magnitude=0.5, date=None,
            source_text="text", confidence=0.8
        )
        assert event.validate() is False

    def test_invalid_impact(self):
        from encoders.event_schema import FinancialEvent
        event = FinancialEvent(
            company="Apple", ticker=None, event_type="earnings",
            description="Some event.", impact="VERY_BAD",
            impact_magnitude=0.5, date=None,
            source_text="text", confidence=0.8
        )
        assert event.validate() is False

    def test_to_dict_and_from_dict_roundtrip(self):
        from encoders.event_schema import FinancialEvent
        event = FinancialEvent(
            company="Tesla", ticker="TSLA", event_type="product_launch",
            description="Tesla launches new model.", impact="positive",
            impact_magnitude=0.6, date="2024-03-01",
            source_text="Tesla launched...", confidence=0.85
        )
        d = event.to_dict()
        recovered = FinancialEvent.from_dict(d)
        assert recovered.company == event.company
        assert recovered.ticker == event.ticker
        assert recovered.impact == event.impact


# ── EventBatch tests ──────────────────────────────────────────────────────────

class TestEventBatch:

    def _make_event(self, company, impact):
        from encoders.event_schema import FinancialEvent
        return FinancialEvent(
            company=company, ticker=None, event_type="earnings",
            description=f"{company} event.", impact=impact,
            impact_magnitude=0.5, date=None,
            source_text="text", confidence=0.8
        )

    def test_add_and_len(self):
        from encoders.event_schema import EventBatch
        batch = EventBatch()
        batch.add(self._make_event("Apple", "negative"))
        batch.add(self._make_event("Tesla", "positive"))
        assert len(batch) == 2

    def test_filter_by_impact(self):
        from encoders.event_schema import EventBatch
        batch = EventBatch()
        batch.add(self._make_event("Apple", "negative"))
        batch.add(self._make_event("Tesla", "positive"))
        batch.add(self._make_event("Meta", "negative"))
        neg = batch.filter_by_impact("negative")
        assert len(neg) == 2

    def test_filter_by_company(self):
        from encoders.event_schema import EventBatch
        batch = EventBatch()
        batch.add(self._make_event("Apple", "negative"))
        batch.add(self._make_event("Tesla", "positive"))
        apple_events = batch.filter_by_company("Apple")
        assert len(apple_events) == 1

    def test_json_roundtrip(self):
        from encoders.event_schema import EventBatch
        batch = EventBatch(source_date="2024-01-15")
        batch.add(self._make_event("Apple", "negative"))
        json_str = batch.to_json()
        recovered = EventBatch.from_json(json_str)
        assert len(recovered) == 1
        assert recovered.events[0].company == "Apple"


# ── EventCache tests ──────────────────────────────────────────────────────────

class TestEventCache:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_and_get(self):
        from encoders.cache import EventCache
        cache = EventCache(cache_dir=self.tmpdir)
        cache.set("some news text", '{"events": []}')
        result = cache.get("some news text")
        assert result == '{"events": []}'

    def test_cache_miss_returns_none(self):
        from encoders.cache import EventCache
        cache = EventCache(cache_dir=self.tmpdir)
        assert cache.get("text not in cache") is None

    def test_cache_size(self):
        from encoders.cache import EventCache
        cache = EventCache(cache_dir=self.tmpdir)
        cache.set("text1", "result1")
        cache.set("text2", "result2")
        assert cache.size() == 2

    def test_cache_clear(self):
        from encoders.cache import EventCache
        cache = EventCache(cache_dir=self.tmpdir)
        cache.set("text1", "result1")
        cache.clear()
        assert cache.size() == 0


# ── NERPostprocessor tests ────────────────────────────────────────────────────

class TestNERPostprocessor:

    def test_resolve_known_ticker_exact(self):
        from encoders.ner_postprocessor import NERPostprocessor
        ner = NERPostprocessor()
        assert ner.resolve_ticker("Apple") == "AAPL"

    def test_resolve_known_ticker_case_insensitive(self):
        from encoders.ner_postprocessor import NERPostprocessor
        ner = NERPostprocessor()
        assert ner.resolve_ticker("apple") == "AAPL"
        assert ner.resolve_ticker("MICROSOFT") == "MSFT"

    def test_resolve_unknown_ticker_returns_none(self):
        from encoders.ner_postprocessor import NERPostprocessor
        ner = NERPostprocessor()
        assert ner.resolve_ticker("SomeRandomCompanyXYZ") is None

    def test_enrich_events_fills_ticker(self):
        from encoders.ner_postprocessor import NERPostprocessor
        from encoders.event_schema import FinancialEvent
        ner = NERPostprocessor()
        event = FinancialEvent(
            company="Apple Inc.", ticker=None, event_type="earnings",
            description="Apple missed earnings.", impact="negative",
            impact_magnitude=0.7, date=None,
            source_text=SAMPLE_NEWS, confidence=0.9
        )
        enriched = ner.enrich_events([event], SAMPLE_NEWS)
        assert enriched[0].ticker == "AAPL"


# ── EventExtractor parse tests (no API calls) ─────────────────────────────────

class TestEventExtractorParsing:
    """Test the parsing logic without making real API calls.
    
    OpenAI client is mocked so no real API key or network is needed.
    """

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Patch openai.OpenAI for ALL tests in this class so __init__ never
        # tries to create a real client.
        self.mock_openai_patcher = patch("encoders.event_extractor.OpenAI")
        self.mock_openai_cls = self.mock_openai_patcher.start()
        self.mock_openai_cls.return_value = MagicMock()

    def teardown_method(self):
        self.mock_openai_patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_extractor(self, use_cache=False):
        from encoders.event_extractor import EventExtractor
        return EventExtractor(
            backend="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            cache_dir=self.tmpdir,
            use_cache=use_cache
        )

    def test_parse_clean_json(self):
        extractor = self._make_extractor()
        events = extractor._parse_response(SAMPLE_EVENTS_JSON, SAMPLE_NEWS)
        assert len(events) == 1
        assert events[0].company == "Apple Inc."
        assert events[0].impact == "negative"

    def test_parse_fenced_json(self):
        """LLM often wraps JSON in ```json ... ``` blocks."""
        extractor = self._make_extractor()
        events = extractor._parse_response(SAMPLE_EVENTS_FENCED, SAMPLE_NEWS)
        assert len(events) == 1
        assert events[0].company == "Apple Inc."

    def test_parse_invalid_json_returns_empty(self):
        extractor = self._make_extractor()
        events = extractor._parse_response("NOT VALID JSON {broken", SAMPLE_NEWS)
        assert events == []

    def test_extract_uses_cache(self):
        """Second call with same text should not call the API."""
        extractor = self._make_extractor(use_cache=True)
        # Prime the cache manually
        extractor.cache.set(SAMPLE_NEWS, SAMPLE_EVENTS_JSON)

        # Patch _call_openai to ensure it is NOT called
        with patch.object(extractor, "_call_openai") as mock_api:
            batch = extractor.extract(SAMPLE_NEWS)
            mock_api.assert_not_called()
        assert len(batch) == 1

    def test_extract_batch(self):
        extractor = self._make_extractor(use_cache=True)
        texts = [SAMPLE_NEWS, "Tesla launched a new vehicle line."]

        # Pre-populate cache so no real API calls happen
        extractor.cache.set(texts[0], SAMPLE_EVENTS_JSON)
        extractor.cache.set(texts[1], json.dumps([{
            "company": "Tesla", "event_type": "product_launch",
            "description": "Tesla new vehicle.", "impact": "positive",
            "impact_magnitude": 0.6, "date": None, "confidence": 0.88
        }]))

        batches = extractor.extract_batch(texts)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
