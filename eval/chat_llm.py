import hashlib
import json
import logging
import os
import os.path as osp
import pickle

from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO)


def calculate_hash(data) -> str:
    string_representation = json.dumps(data, sort_keys=True)

    sha256 = hashlib.sha256()
    sha256.update(string_representation.encode("utf-8"))
    return sha256.hexdigest()


class ChatLLM:
    def __init__(self, chat_name, model_name: str = "gpt-4", use_cache=False):
        self.client = AzureOpenAI(api_version="2023-12-01-preview")

        self.model_name = model_name
        self.chat_name = chat_name
        self.use_cache = use_cache
        if use_cache:
            self.init_cache()

    def init_cache(self):
        self.cache_file_name = f"cache_chatllm_{self.chat_name}.pkl"
        self.cache = self.load_cache()

    def load_cache(self):
        if osp.isfile(self.cache_file_name):
            with open(self.cache_file_name, "rb") as f:
                return pickle.load(f)
        return dict()

    def add_to_cache(self, chat_messages, response):
        if not self.use_cache:
            return
        hash_key = calculate_hash(chat_messages)
        self.cache[hash_key] = response
        with open(self.cache_file_name, "wb") as f:
            pickle.dump(self.cache, f)

    def _chat(self, chat_messages, stream=False, temperature=0):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=temperature,
            stream=stream,
            max_tokens=2048,
        )
        if stream:
            for chunk in response:
                if len(chunk.choices) and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            message = response.choices[0].message
            if hasattr(message, "content"):
                yield message.content
            else:
                yield ""

    def chat(self, chat_messages, stream=False, temperature=0):
        if self.use_cache:
            hash_key = calculate_hash(chat_messages)
            if hash_key in self.cache:
                logging.info(f"Found in cache:\n{self.cache[hash_key]}")
                return self.cache[hash_key]

        final_answer = ""
        for res in self._chat(chat_messages, stream, temperature=temperature):
            print(res, end="", flush=True)
            final_answer += res
        print()
        self.add_to_cache(chat_messages, final_answer)
        return final_answer
