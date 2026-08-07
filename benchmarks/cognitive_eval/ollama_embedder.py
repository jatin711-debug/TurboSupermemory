"""SentenceTransformer-compatible embeddings served by local Ollama."""

import numpy as np


class OllamaEmbedder:
    def __init__(
        self,
        model="qllama/bge-large-en-v1.5:f16",
        dim=1024,
        host="http://localhost:11434",
        batch=128,
    ):
        import ollama

        self._client = ollama.Client(host=host)
        self.model = model
        self._dim = dim
        self.batch = batch
        self.calls = 0
        self._cache = {}

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **_kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        normalized = [text if text and text.strip() else " " for text in items]
        missing = list(dict.fromkeys(text for text in normalized if text not in self._cache))
        for start in range(0, len(missing), self.batch):
            chunk = missing[start:start + self.batch]
            self.calls += 1
            response = self._client.embed(model=self.model, input=chunk)
            for text, vector in zip(chunk, response.embeddings):
                array = np.asarray(vector, dtype=np.float32)
                if array.shape != (self._dim,):
                    raise ValueError(
                        f"Ollama embedding dimension {array.shape} does not match {self._dim}"
                    )
                self._cache[text] = array
        output = np.vstack([self._cache[text] for text in normalized]).astype(np.float32)
        return output[0] if single else output
