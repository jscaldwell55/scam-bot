from collections import defaultdict
from typing import List, Dict, Any

# Default structure for each user's session memory
def default_user_memory():
    return {
        "messages": [],          # conversation history (list of dicts)
        "trust_score": 0.0,      # can be used to track manipulation escalation
        "emotional_state": "neutral",  # placeholder: "nostalgic", "vulnerable", etc.
        "facts": {},             # fake facts Janet invents about the user's life
    }

# Master memory object
memory: Dict[str, Dict[str, Any]] = defaultdict(default_user_memory)

# Constants
MAX_HISTORY = 12  # keep last 12 messages (user/assistant pairs)
