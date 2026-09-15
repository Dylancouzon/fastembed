import pytest
import numpy as np

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import last_token_pooling
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding


def test_text_list_supported_models():
    for model_type in [
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        LateInteractionTextEmbedding,
    ]:
        supported_models = model_type.list_supported_models()
        assert isinstance(supported_models, list)
        description = supported_models[0]
        assert isinstance(description, dict)

        assert "model" in description and description["model"]
        if model_type != SparseTextEmbedding:
            assert "dim" in description and description["dim"]
        assert "license" in description and description["license"]
        assert "size_in_GB" in description and description["size_in_GB"]
        assert "model_file" in description and description["model_file"]
        assert "sources" in description and description["sources"]
        assert "hf" in description["sources"] or "url" in description["sources"]


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_pooled_post_processing_returns_graph_dtype(dtype):
    """Pooling accumulates in float64 because the integer mask promotes the product.

    The returned embedding must still carry the dtype the ONNX graph produced, like every
    non-pooled path. Narrowing before normalize() would overflow float16's squared sum.
    """
    token_embeddings = np.array([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]], dtype=dtype)
    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)
    output = OnnxOutputContext(model_output=token_embeddings, attention_mask=attention_mask)

    pooled = PooledEmbedding._post_process_onnx_output(
        PooledEmbedding.__new__(PooledEmbedding), output
    )
    assert pooled.dtype == dtype
    assert np.allclose(np.asarray(pooled, dtype=np.float64), [[2.0, 3.0]])

    normalized = PooledNormalizedEmbedding._post_process_onnx_output(
        PooledNormalizedEmbedding.__new__(PooledNormalizedEmbedding), output
    )
    assert normalized.dtype == dtype
    tolerance = 1e-3 if dtype is np.float16 else 1e-6
    assert np.allclose(
        np.linalg.norm(np.asarray(normalized, dtype=np.float64), axis=1), 1.0, atol=tolerance
    )


def test_pooled_normalized_does_not_overflow_float16():
    """A float16 graph output large enough to overflow its own squared sum must still
    normalize: the regression that motivated this was a vector of exact zeros."""
    token_embeddings = np.full((1, 2, 1024), 10.0, dtype=np.float16)
    attention_mask = np.ones((1, 2), dtype=np.int64)
    output = OnnxOutputContext(model_output=token_embeddings, attention_mask=attention_mask)

    normalized = PooledNormalizedEmbedding._post_process_onnx_output(
        PooledNormalizedEmbedding.__new__(PooledNormalizedEmbedding), output
    )
    assert normalized.dtype == np.float16
    assert not np.all(normalized == 0)
    assert np.allclose(np.linalg.norm(np.asarray(normalized, dtype=np.float64), axis=1), 1.0, atol=1e-3)


def test_last_token_pooling():
    token_embeddings = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],  # 2 real tokens, then padding
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_last_token_pooling_with_left_padding():
    token_embeddings = np.array(
        [
            [[9.0, 9.0], [9.0, 9.0], [1.0, 1.0], [2.0, 2.0]],  # padding, then 2 real tokens
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])
