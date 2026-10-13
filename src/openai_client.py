"""openai_client.py — thin wrapper for embeddings and chat completions."""

import os, time
import numpy as np
from typing import List, Optional
from openai import OpenAI

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=key)
    return _client


def embed_texts(
    texts: List[str], model="text-embedding-3-small", batch_size=512
) -> np.ndarray:
    client = get_client()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(input=batch, model=model)
                all_embs.extend([item.embedding for item in resp.data])
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2**attempt)
                else:
                    raise
    return np.array(all_embs, dtype=np.float32)


def embed_single(text: str, model="text-embedding-3-small") -> np.ndarray:
    return embed_texts([text], model=model)[0]


def chat_completion(
    prompt: str,
    system: str = "",
    model="gpt-4o-mini",
    max_tokens=200,
    temperature=0.2,
    json_mode=False,
) -> str:
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise
