"""Shared pytest fixtures and lightweight stubs for optional dependencies."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _maybe_stub_openai() -> None:
    if importlib.util.find_spec('openai'):
        return

    module = types.ModuleType('openai')

    class DummyOpenAIError(Exception):
        """Placeholder exception to satisfy type checking."""

    class _Completions:
        def create(self, *args, **kwargs):  # noqa: D401, ANN001 - simple stub
            message = SimpleNamespace(content='{}')
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class DummyOpenAI:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN001
            self.chat = _Chat()

    module.OpenAI = DummyOpenAI
    module.OpenAIError = DummyOpenAIError
    sys.modules['openai'] = module


def _maybe_stub_qdrant_client() -> None:
    if importlib.util.find_spec('qdrant_client'):
        return

    root = types.ModuleType('qdrant_client')

    class DummyQdrantClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, D401
            self.kwargs = kwargs

        def recreate_collection(self, *args, **kwargs) -> None:  # noqa: ANN001
            return None

        def upload_points(self, *args, **kwargs) -> None:  # noqa: ANN001
            return None

        def query_points(self, *args, **kwargs):  # noqa: ANN001, D401
            return SimpleNamespace(points=[])

    root.QdrantClient = DummyQdrantClient
    sys.modules['qdrant_client'] = root

    models = types.ModuleType('qdrant_client.models')

    class Distance:
        COSINE = 'cosine'

    class VectorParams:
        def __init__(self, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id: int, vector, payload) -> None:  # noqa: ANN001
            self.id = id
            self.vector = vector
            self.payload = payload

    models.Distance = Distance
    models.VectorParams = VectorParams
    models.PointStruct = PointStruct
    sys.modules['qdrant_client.models'] = models


def _maybe_stub_gradio_client() -> None:
    if importlib.util.find_spec('gradio_client'):
        return

    module = types.ModuleType('gradio_client')

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, D401
            self.kwargs = kwargs

        def predict(self, *args, **kwargs):  # noqa: ANN001, D401
            return {}

    module.Client = DummyClient
    sys.modules['gradio_client'] = module


_maybe_stub_openai()
_maybe_stub_qdrant_client()
_maybe_stub_gradio_client()
