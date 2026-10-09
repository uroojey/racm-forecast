import os
import hashlib
from typing import Optional

class EventCache:
    """
    Disk-based cache using JSON files. Keyed by hash of input text.
    Prevents redundant API calls.
    """
    def __init__(self, cache_dir: str = ".cache/events"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _key(self, text: str) -> str:
        """SHA256 hash of the input text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[str]:
        """Return cached JSON string if exists, else None."""
        key = self._key(text)
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def set(self, text: str, result: str) -> None:
        """Save result JSON string to cache."""
        key = self._key(text)
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result)
    
    def clear(self) -> None:
        """Delete all cached files."""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, filename))
    
    def size(self) -> int:
        """Return number of cached entries."""
        if not os.path.exists(self.cache_dir):
            return 0
        return len([f for f in os.listdir(self.cache_dir) if f.endswith(".json")])
