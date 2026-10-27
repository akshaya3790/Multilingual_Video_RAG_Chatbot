"""Check the configured inference connection without printing credentials."""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def main():
    load_dotenv(Path(__file__).with_name('.env'))
    token = os.getenv('HF_TOKEN', '')
    if not token:
        print('HF_TOKEN is missing from .env.')
        return 1
    try:
        response = InferenceClient(api_key=token, timeout=45).chat.completions.create(
            model=os.getenv('HF_LLM_MODEL', 'openai/gpt-oss-20b'),
            messages=[{'role': 'user', 'content': 'Reply with OK.'}],
            max_tokens=512,
        )
        print('Inference reply:', response.choices[0].message.content)
        return 0
    except Exception as exc:
        print('Error type:', type(exc).__name__)
        print('HTTP status:', getattr(getattr(exc, 'response', None), 'status_code', None))
        print('Detail:', str(exc).replace(token, '[REDACTED]'))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
