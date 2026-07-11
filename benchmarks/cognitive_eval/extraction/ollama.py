"""Ollama-based fact extractor for benchmark compatibility.

Uses a local LLM (via Ollama) to extract atomic facts from conversation
messages. This matches Mem0's single-pass extraction approach.

Requirements:
    pip install ollama
    # And have Ollama running with a model pulled:
    ollama pull llama3.2:3b  # or your preferred model

Usage:
    extractor = OllamaExtractor(model="llama3.2:3b")
    facts = extractor.extract_facts("I just moved to San Francisco yesterday.")
    # Returns: ["User moved to San Francisco", "Move happened yesterday"]
"""

import json
import logging
import re
from typing import List, Optional

logger = logging.getLogger("cognitive_eval.extraction.ollama")


class OllamaExtractor:
    """Extract atomic facts from text using a local LLM via Ollama."""
    
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = None
        
        try:
            import ollama
            self._client = ollama.Client(host=host)
            logger.info("Ollama client initialized (model=%s, host=%s)", model, host)
        except ImportError:
            logger.error(
                "ollama package not installed. Install with: pip install ollama\n"
                "Also ensure Ollama is running: ollama serve"
            )
            raise
    
    def _build_prompt(self, message: str, context: Optional[List[str]] = None) -> str:
        """Build the extraction prompt."""
        context_str = ""
        if context:
            context_str = "\n".join(f"  - {c}" for c in context[-3:])
            context_str = f"\nRecent context:\n{context_str}\n"
        
        return f"""Extract atomic facts from the following message. 

An atomic fact is a self-contained piece of information that can stand alone.
Break complex statements into multiple simple facts.
Preserve temporal information (when, before, after, now, yesterday, etc.).

{context_str}
Message to extract from:
"{message}"

Return ONLY a JSON object in this exact format:
{{"facts": ["fact 1", "fact 2", "fact 3"]}}

If no facts can be extracted, return: {{"facts": []}}
"""
    
    def _parse_response(self, response_text: str) -> List[str]:
        """Parse the LLM response to extract facts."""
        # Try to find JSON in the response
        # The model might wrap it in markdown or add extra text
        
        # Look for JSON block
        json_match = re.search(r'\{.*"facts"\s*:\s*\[.*\].*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                facts = data.get("facts", [])
                if isinstance(facts, list):
                    return [f.strip() for f in facts if f.strip()]
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for bullet points or numbered lists
        facts = []
        for line in response_text.split('\n'):
            line = line.strip()
            # Remove bullet markers and numbers
            line = re.sub(r'^[\s]*[-*•]\s+', '', line)
            line = re.sub(r'^\d+\.\s+', '', line)
            if line and len(line) > 10:  # Minimum length for a fact
                facts.append(line)
        
        return facts
    
    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        """Extract atomic facts from a message.
        
        Args:
            message: The message text to extract facts from
            context: Optional list of recent messages for context
            
        Returns:
            List of atomic fact strings
        """
        if not message or not message.strip():
            return []
        
        if self._client is None:
            logger.warning("Ollama client not available, returning message as single fact")
            return [message.strip()]
        
        prompt = self._build_prompt(message, context)
        
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            
            response_text = response.message.content
            facts = self._parse_response(response_text)
            
            if not facts:
                # Fallback: use the message itself as a single fact
                facts = [message.strip()]
            
            logger.debug("Extracted %d facts from message: %s", len(facts), message[:50])
            return facts
            
        except Exception as e:
            logger.error("Ollama extraction failed: %s. Falling back to raw message.", e)
            return [message.strip()]
    
    def extract_facts_batch(
        self,
        messages: List[str],
        contexts: Optional[List[List[str]]] = None,
    ) -> List[List[str]]:
        """Extract facts from multiple messages.
        
        Args:
            messages: List of messages to extract from
            contexts: Optional list of context lists (one per message)
            
        Returns:
            List of fact lists (one per message)
        """
        results = []
        for i, message in enumerate(messages):
            context = contexts[i] if contexts and i < len(contexts) else None
            facts = self.extract_facts(message, context)
            results.append(facts)
        return results
    
    def health_check(self) -> bool:
        """True only if the Ollama server is reachable AND this model is pulled
        (a reachable server with a missing model 404s at call time)."""
        try:
            r = self._client.list()
            raw = r.models if hasattr(r, "models") else r.get("models", [])
            names = []
            for m in raw:
                names.append(getattr(m, "model", None) or (m.get("model") or m.get("name")
                             if isinstance(m, dict) else None))
            ok = any(n and (n == self.model or n.startswith(self.model)) for n in names)
            if not ok:
                logger.warning("Ollama model '%s' not found among %s", self.model, names)
            return ok
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test the extractor
    extractor = OllamaExtractor()
    
    test_messages = [
        "I just moved to San Francisco yesterday. The weather is much better here than in New York.",
        "I love the coffee shops in the Mission District. I go to Ritual Coffee every morning.",
        "Actually, I changed my mind. I'm moving back to New York next month because the rent is too high.",
    ]
    
    print("Testing Ollama fact extraction...")
    print("=" * 60)
    
    context = []
    for msg in test_messages:
        print(f"\nMessage: {msg}")
        facts = extractor.extract_facts(msg, context)
        print("Facts:")
        for fact in facts:
            print(f"  - {fact}")
        context.append(msg)
        print("-" * 60)
