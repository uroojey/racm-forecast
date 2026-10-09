class PromptTemplates:
    SYSTEM_PROMPT = (
        "You are a financial news analyst. Your task is to extract structured financial "
        "events from news text. Always output your response as a valid JSON array of objects."
    )

    @classmethod
    def extraction_prompt(cls, text: str) -> str:
        return f"""Extract financial events from the provided text.

Output ONLY a valid JSON array of event objects. Each object must have these exact keys:
- company: string (company name)
- event_type: string (one of: earnings, merger, lawsuit, leadership_change, product_launch, regulatory, macro, dividend, other)
- description: string (1-line description)
- impact: string (positive, negative, neutral)
- impact_magnitude: float (0.0 to 1.0)
- date: string (YYYY-MM-DD or null)
- confidence: float (0.0 to 1.0)

Few-shot examples:
Text: "Apple reported a 10% drop in Q3 earnings, missing analyst estimates."
JSON: [{{"company": "Apple", "event_type": "earnings", "description": "10% drop in Q3 earnings missing estimates", "impact": "negative", "impact_magnitude": 0.7, "date": null, "confidence": 0.9}}]

Text: "Tesla announced the launch of its new Model 2, expected to boost sales next year."
JSON: [{{"company": "Tesla", "event_type": "product_launch", "description": "Launch of new Model 2", "impact": "positive", "impact_magnitude": 0.6, "date": null, "confidence": 0.8}}]

Text: "The Federal Reserve raised interest rates by 25 basis points today."
JSON: [{{"company": "Federal Reserve", "event_type": "macro", "description": "Federal Reserve raised interest rates by 25 basis points", "impact": "negative", "impact_magnitude": 0.8, "date": null, "confidence": 0.9}}]

Text to analyze:
{text}
"""

    @classmethod
    def validation_prompt(cls, event_json: str) -> str:
        return f"""The following JSON is invalid or malformed. Fix it and return ONLY a valid JSON array of event objects.
JSON:
{event_json}
"""
