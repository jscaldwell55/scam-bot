# memory_store.py
from typing import Dict, List, Any

# Simple in-memory storage for conversation history
memory: Dict[str, Dict[str, Any]] = {}

# Constants
MAX_HISTORY = 20  # Keep last 20 messages for better context