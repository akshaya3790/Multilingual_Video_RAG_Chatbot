"""Explicit, user-approved classifier download; never called by the app.

Superseded by download_models.py, which fetches all local verifier models at pinned revisions.
"""
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

if __name__ == '__main__':
    repo = 'cross-encoder/nli-deberta-v3-xsmall'
    info = HfApi().model_info(repo)
    files = HfApi().list_repo_files(repo, revision=info.sha)
    if 'model.safetensors' not in files:
        raise RuntimeError('Safe weights unavailable; download stopped.')
    target = Path(__file__).parent / 'local_models' / 'nli-deberta-v3-xsmall'
    snapshot_download(repo, revision=info.sha, local_dir=target,
        allow_patterns=['*.json', '*.model', '*.txt', 'model.safetensors'],
        ignore_patterns=['onnx/*', 'openvino/*'])
    print('Downloaded pinned revision:', info.sha)
    print('Model directory:', target.resolve())
