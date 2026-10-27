"""Download the optional local verifier models at pinned revisions.

Only GROUNDING_CHECKER=local_ensemble needs them. The small, reviewed HHEM code
and config files are kept in the repository; this fetches the large weights and
tokenizers into local_models/ and checks them. Run: python download_models.py
"""
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from grounding_ensemble import FOUNDATION_REVISION, verified_sources


ROOT = Path(__file__).parent / 'local_models'
HHEM_REPO = 'vectara/hallucination_evaluation_model'
HHEM_REVISION = 'de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8'
HHEM_WEIGHTS_SHA256 = '634de18a38cf1e991c1acd0f7a9e0d30f7ea187fba42bb4798f862d3edd31e72'
NLI_REPO = 'cross-encoder/nli-deberta-v3-xsmall'
NLI_REVISION = 'a150876415327c80daeff35ca6f68f5ed8cf5c24'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    hhem = ROOT / 'hhem-2.1-open'
    # Refuse to add weights next to code that is missing or changed.
    verified_sources(hhem)
    weights = hhem / 'model.safetensors'
    if not weights.is_file() or sha256(weights) != HHEM_WEIGHTS_SHA256:
        hf_hub_download(HHEM_REPO, 'model.safetensors', revision=HHEM_REVISION, local_dir=hhem)
        if sha256(weights) != HHEM_WEIGHTS_SHA256:
            raise SystemExit('Downloaded HHEM weights do not match the pinned hash; not using them.')
    snapshot_download('google/flan-t5-base', revision=FOUNDATION_REVISION,
        cache_dir=ROOT / 'hhem_hub_cache',
        allow_patterns=['config.json', 'special_tokens_map.json', 'spiece.model',
                        'tokenizer.json', 'tokenizer_config.json'])
    snapshot_download(NLI_REPO, revision=NLI_REVISION, local_dir=ROOT / 'nli-deberta-v3-xsmall',
        allow_patterns=['*.json', '*.model', 'model.safetensors'])
    print('Local verifier models are ready in', ROOT.resolve())


if __name__ == '__main__':
    main()
