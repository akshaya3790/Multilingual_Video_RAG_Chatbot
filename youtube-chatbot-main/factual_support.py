"""Independent English NLI support classification; no generation or network I/O."""
from dataclasses import dataclass
from pathlib import Path
import threading


MODEL_DIRECTORY = Path(__file__).parent / 'local_models' / 'nli-deberta-v3-xsmall'


class SupportClassifierError(RuntimeError):
    """A technical failure, never evidence that the transcript lacks an answer."""


@dataclass(frozen=True)
class SupportDecision:
    label: str
    scores: dict[str, float]


class FactualSupportClassifier:
    """CPU-only, lazy, local-files-only inference with explicit length checks.

    Probabilities are model scores, not calibrated correctness probabilities.
    Each call checks one passage/claim pair; question relevance is not assessed.
    """

    def __init__(self, model_directory=MODEL_DIRECTORY):
        self.model_directory = Path(model_directory)
        self._model = self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        if not (self.model_directory / 'model.safetensors').is_file():
            raise SupportClassifierError('Local support classifier files are unavailable.')
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_directory,
                local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_directory,
                local_files_only=True, trust_remote_code=False, use_safetensors=True).to('cpu').eval()
            labels = {int(i): str(label).lower() for i, label in model.config.id2label.items()}
            if set(labels.values()) != {'entailment', 'contradiction', 'neutral'}:
                raise SupportClassifierError('Unsupported classifier label mapping.')
            self._labels = labels
            self._tokenizer, self._model = tokenizer, model
        except SupportClassifierError:
            raise
        except Exception:
            raise SupportClassifierError('Local support classifier could not load.') from None

    def classify(self, passage, claim):
        if not isinstance(passage, str) or not passage.strip() or not isinstance(claim, str) or not claim.strip():
            raise SupportClassifierError('A nonempty passage and claim are required.')
        with self._lock:
            self._load()
            try:
                import torch
                inputs = self._tokenizer(passage, claim, return_tensors='pt', truncation=False)
                limit = min(512, self._model.config.max_position_embeddings)
                if inputs['input_ids'].shape[1] > limit:
                    return SupportDecision('uncertain', {})
                with torch.inference_mode():
                    values = self._model(**inputs).logits.softmax(dim=-1)[0]
                if not torch.isfinite(values).all():
                    raise SupportClassifierError('Nonfinite classifier scores.')
                scores = {self._labels[i]: float(value) for i, value in enumerate(values)}
                label = max(scores, key=scores.get)
                return SupportDecision(label, scores)
            except SupportClassifierError:
                raise
            except Exception:
                raise SupportClassifierError('Local support classification failed.') from None


classifier = FactualSupportClassifier()
