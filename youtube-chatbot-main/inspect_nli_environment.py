"""Read-only local resource inspection for the independent classifier (any OS)."""
from factual_support import MODEL_DIRECTORY
from system_memory import memory_mib


if __name__ == '__main__':
    total, available = memory_mib()
    print('RAM MiB total/available:', total, available)
    print('Classifier model folder:', MODEL_DIRECTORY)
    print('Classifier weights present:', (MODEL_DIRECTORY / 'model.safetensors').is_file())
