from openai import OpenAI
import os
import logging

# Lazy initialization of OpenAI client
_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "You need to set OpenAI API key. Set OPENAI_API_KEY environment variable or pass it explicitly.")
        _client = OpenAI(api_key=api_key)
    return _client

def openai_request(messages, model='gpt-3.5-turbo', temperature=1, top_n=1, max_trials=100):
    client = get_openai_client()

    for _ in range(max_trials):
        try:
            results = client.completions.create(
                model=model, messages=messages, temperature=temperature, n=top_n
            )

            assert len(results.choices) == top_n
            return results
        except Exception as e:
            logging.warning("OpenAI API call failed. Retrying...")
