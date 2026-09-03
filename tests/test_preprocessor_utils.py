import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from fastembed.common.preprocessor_utils import load_tokenizer


def _write_model_dir(tmp_path, padding, model_max_length=20):
    """A minimal model directory that load_tokenizer accepts."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    if padding is not None:
        tokenizer.enable_padding(**padding)
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 0}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": model_max_length, "pad_token": "[PAD]"})
    )
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "[PAD]", "unk_token": "[UNK]"})
    )
    return tmp_path


def test_fixed_padding_shorter_than_truncation_does_not_produce_ragged_batches(tmp_path):
    """A fixed padding length below the truncation limit leaves longer inputs neither padded
    nor truncated, so batches come out ragged and np.array() raises. Regression test for
    thenlper/gte-base (Fixed 128 against truncation 512), which worked in 0.7.4."""
    model_dir = _write_model_dir(
        tmp_path, padding={"length": 4, "pad_id": 0, "pad_token": "[PAD]"}, model_max_length=20
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding is not None
    assert tokenizer.padding["length"] is None, "padding must be dynamic, not fixed"

    encoded = tokenizer.encode_batch(["a b", "a b a b a b a b"])
    assert len({len(e.ids) for e in encoded}) == 1, "batch rows must be equal length"


def test_padding_direction_is_preserved(tmp_path):
    """Left-padding models (e.g. colmodernvbert) must keep their direction."""
    model_dir = _write_model_dir(
        tmp_path,
        padding={"direction": "left", "length": 4, "pad_id": 0, "pad_token": "[PAD]"},
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["direction"] == "left"
    assert tokenizer.padding["length"] is None


def test_truncation_still_applies(tmp_path):
    model_dir = _write_model_dir(tmp_path, padding=None, model_max_length=3)

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.truncation["max_length"] == 3
    assert len(tokenizer.encode("a b a b a b").ids) == 3
