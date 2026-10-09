import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List

@dataclass
class FinancialEvent:
    company: str
    ticker: Optional[str]
    event_type: str
    description: str
    impact: str
    impact_magnitude: float
    date: Optional[str]
    source_text: str
    confidence: float

    VALID_EVENT_TYPES = ["earnings", "merger", "lawsuit", "leadership_change",
                         "product_launch", "regulatory", "macro", "dividend", "other"]
    VALID_IMPACTS = ["positive", "negative", "neutral"]

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "FinancialEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def validate(self) -> bool:
        """Returns True if the event has valid fields."""
        return (
            bool(self.company) and
            bool(self.description) and
            self.event_type in self.VALID_EVENT_TYPES and
            self.impact in self.VALID_IMPACTS and
            0.0 <= self.impact_magnitude <= 1.0 and
            0.0 <= self.confidence <= 1.0
        )

@dataclass
class EventBatch:
    events: List[FinancialEvent] = field(default_factory=list)
    source_date: Optional[str] = None
    source_url: Optional[str] = None
    
    def add(self, event: FinancialEvent):
        self.events.append(event)
        
    def filter_by_impact(self, impact: str) -> List[FinancialEvent]:
        return [e for e in self.events if e.impact == impact]
        
    def filter_by_company(self, company: str) -> List[FinancialEvent]:
        return [e for e in self.events if e.company.lower() in company.lower() or company.lower() in e.company.lower()]
        
    def to_json(self) -> str:
        d = {
            "events": [e.to_dict() for e in self.events],
            "source_date": self.source_date,
            "source_url": self.source_url
        }
        return json.dumps(d)
        
    @classmethod
    def from_json(cls, json_str: str) -> "EventBatch":
        d = json.loads(json_str)
        events = [FinancialEvent.from_dict(ed) for ed in d.get("events", [])]
        return cls(events=events, source_date=d.get("source_date"), source_url=d.get("source_url"))
        
    def __len__(self):
        return len(self.events)
