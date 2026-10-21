"""An OpenAI-client-shaped chat model that runs in this process.

``tsm.verification.LLMVerifier`` talks to ``client.chat.completions.create``.
This gives it a causal LM from the Hugging Face cache instead of a server, so a
verifier can be evaluated with no network, no key and no llama.cpp:

    from local_chat import LocalChatClient
    LLMVerifier(model="LiquidAI/LFM2.5-1.2B-Instruct", client=LocalChatClient("LiquidAI/LFM2.5-1.2B-Instruct"))

It is for measuring, not serving: one request at a time, greedy decoding,
no batching. A GGUF model such as Ternary-Bonsai-2 is served by the PrismML
llama.cpp fork and reached through ``LLMVerifier(base_url=...)`` instead.
"""

import threading
from types import SimpleNamespace


class LocalChatClient:
    def __init__(self, model_name, device=None, dtype="bfloat16", max_new_tokens=400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and dtype == "bfloat16":
            dtype = "float32"
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.calls = 0
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype)).to(device).eval()
        self._torch = torch
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model=None, messages=None, temperature=0.0, max_tokens=None, **_ignored):
        torch = self._torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self._lock, torch.no_grad():
            self.calls += 1
            out = self._model.generate(
                **batch, max_new_tokens=int(max_tokens or self.max_new_tokens), do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id)
        text = self._tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
