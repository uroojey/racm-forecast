import logging
from typing import Optional, Dict, List
from .event_schema import FinancialEvent

logger = logging.getLogger(__name__)

class NERPostprocessor:
    """
    Uses spaCy to extract organization names from text and
    resolves them to stock tickers using a lookup dictionary.
    """
    def __init__(self, ticker_map: Optional[Dict[str, str]] = None):
        """
        ticker_map: dict like {"Apple": "AAPL", "Microsoft": "MSFT", ...}
        Falls back to a built-in small default map if none provided.
        Loads spaCy model lazily (en_core_web_sm).
        """
        self.DEFAULT_TICKER_MAP = {
            "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
            "Alphabet": "GOOGL", "Amazon": "AMZN", "Meta": "META",
            "Tesla": "TSLA", "Nvidia": "NVDA", "Netflix": "NFLX",
            "JPMorgan": "JPM", "Goldman Sachs": "GS", "Bank of America": "BAC",
            "Berkshire Hathaway": "BRK.B", "Walmart": "WMT", "Johnson & Johnson": "JNJ",
            "Pfizer": "PFE", "Exxon": "XOM", "Chevron": "CVX",
            "Boeing": "BA", "Coca-Cola": "KO", "PepsiCo": "PEP",
            "Intel": "INTC", "AMD": "AMD", "Qualcomm": "QCOM",
            "Salesforce": "CRM", "Adobe": "ADBE", "Oracle": "ORCL",
            "IBM": "IBM", "Cisco": "CSCO", "Disney": "DIS"
        }
        self.ticker_map = ticker_map or self.DEFAULT_TICKER_MAP
        self.nlp = None

    def _load_spacy(self):
        """Lazily load spaCy model, with graceful fallback if not installed."""
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except ImportError:
                logger.warning("spaCy not installed. NER features will be disabled.")
                self.nlp = False
            except OSError:
                logger.warning("en_core_web_sm not found. Try `python -m spacy download en_core_web_sm`.")
                self.nlp = False

    def resolve_ticker(self, company_name: str) -> Optional[str]:
        """Look up ticker for a company name (case-insensitive partial match)."""
        for name, ticker in self.ticker_map.items():
            if name.lower() in company_name.lower() or company_name.lower() in name.lower():
                return ticker
        return None
    
    def extract_orgs(self, text: str) -> List[str]:
        """Use spaCy NER to extract ORG entities from text."""
        self._load_spacy()
        if not self.nlp:
            return []
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    
    def enrich_events(self, events: List[FinancialEvent], source_text: str) -> List[FinancialEvent]:
        """
        For each event without a ticker, try to resolve ticker from company name.
        Also extract ORGs from source_text and try to match.
        Returns enriched events.
        """
        orgs = self.extract_orgs(source_text)
        
        for event in events:
            if not event.ticker:
                ticker = self.resolve_ticker(event.company)
                
                if not ticker:
                    for org in orgs:
                        ticker = self.resolve_ticker(org)
                        if ticker:
                            break
                            
                if ticker:
                    event.ticker = ticker
                    
        return events
