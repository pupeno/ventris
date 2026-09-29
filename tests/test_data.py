import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import ventris.data as data_module
from ventris.data import SEQUENCE_LENGTH, load_prepared_data, prepare_data, train_tokenizer


def test_missing_tokenizer_error_names_the_artifact_and_command(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"tokenizer.*scripts/train_tokenizer\.py"):
        data_module.load_tokenizer()


def test_trained_tokenizer_loads_with_transformers(tmp_path, monkeypatch):
    corpus = Dataset.from_dict({"text": ["a little text", "and some more text"]})
    default_dir = tmp_path / "live"
    default_dir.mkdir()
    (default_dir / "tokenizer.json").write_text("existing tokenizer")
    monkeypatch.setattr(data_module, "DATA_DIR", default_dir)
    monkeypatch.setattr(data_module, "VOCAB_SIZE", 257)
    monkeypatch.setattr(
        data_module,
        "_corpus_splits",
        lambda: DatasetDict({"train": corpus, "validation": corpus}),
    )

    path = train_tokenizer(tmp_path / "candidate")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    assert path == tmp_path / "candidate"
    assert (default_dir / "tokenizer.json").read_text() == "existing tokenizer"
    assert tokenizer.encode("a little text", add_special_tokens=False)
    assert tokenizer.eos_token_id == 256
    assert tokenizer.model_max_length == SEQUENCE_LENGTH
    assert data_module.load_tokenizer(path).encode("a little text", add_special_tokens=False) == (
        tokenizer.encode("a little text", add_special_tokens=False)
    )


def test_tokenizer_training_excludes_validation_documents(tmp_path, monkeypatch):
    training = Dataset.from_dict({"text": ["ab" * 300]})
    validation = Dataset.from_dict({"text": ["z" * 600]})
    monkeypatch.setattr(data_module, "VOCAB_SIZE", 280)
    monkeypatch.setattr(
        data_module,
        "_corpus_splits",
        lambda: DatasetDict({"train": training, "validation": validation}),
    )

    tokenizer = data_module.load_tokenizer(train_tokenizer(tmp_path))

    assert "ab" in tokenizer.get_vocab()
    assert "zz" not in tokenizer.get_vocab()


@pytest.fixture
def prepared_data_environment(tmp_path, monkeypatch):
    corpus = Dataset.from_dict(
        {
            "text": [
                "a " * 600,
                "b " * 600,
                "a " * 600,
                "b " * 600,
            ]
        }
    )
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|endoftext|>").save_pretrained(
        tmp_path
    )

    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "VALIDATION_FRACTION", 0.5)
    monkeypatch.setattr(data_module, "_load_corpus", lambda: corpus)
    return tmp_path


def test_prepare_data_builds_pytorch_batches_across_documents(
    prepared_data_environment, monkeypatch
):
    tmp_path = prepared_data_environment

    data = prepare_data()
    sequences = data["train"][:1]["input_ids"]

    assert set(data) == {"train", "validation"}
    assert len(data["train"]) == len(data["validation"]) == 1
    assert data["train"].column_names == ["input_ids"]
    assert sequences.shape == (1, SEQUENCE_LENGTH + 1)
    assert data_module.load_tokenizer().eos_token_id in sequences
    assert (tmp_path / "prepared" / ".complete").is_file()
    assert not list(tmp_path.glob(".packing-*"))
    assert len(prepare_data()["train"]) == 1

    monkeypatch.setattr(
        data_module,
        "_corpus_splits",
        lambda: (_ for _ in ()).throw(AssertionError("corpus was regenerated")),
    )
    assert len(load_prepared_data()["train"]) == 1


def test_failed_repreparation_marks_data_incomplete(prepared_data_environment, monkeypatch):
    tmp_path = prepared_data_environment
    prepare_data()

    monkeypatch.setattr(
        DatasetDict,
        "save_to_disk",
        lambda self, path: (_ for _ in ()).throw(OSError("copy failed")),
    )
    with pytest.raises(OSError, match="copy failed"):
        prepare_data()
    assert not (tmp_path / "prepared" / ".complete").exists()
    assert not list(tmp_path.glob(".packing-*"))
    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        load_prepared_data()


def test_prepare_data_preserves_document_boundaries_and_overlapping_labels(tmp_path, monkeypatch):
    vocabulary = {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token=data_module.EOS_TEXT
    ).save_pretrained(tmp_path)
    documents = Dataset.from_dict({"text": ["a b", "c d", "e f"]})
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "SEQUENCE_LENGTH", 3)
    monkeypatch.setattr(data_module, "DOCUMENT_BATCH_SIZE", 1)
    monkeypatch.setattr(
        data_module,
        "_corpus_splits",
        lambda: DatasetDict({"train": documents, "validation": documents}),
    )

    prepared = prepare_data()

    assert prepared["train"][:]["input_ids"].tolist() == [[1, 2, 7, 3], [3, 4, 7, 5]]


def test_missing_prepared_data_exits_without_generating_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        data_module,
        "_corpus_splits",
        lambda: (_ for _ in ()).throw(AssertionError("corpus was generated")),
    )

    with pytest.raises(
        FileNotFoundError, match=r"prepared training data.*scripts/prepare_data\.py"
    ):
        load_prepared_data()


def test_incomplete_prepared_data_is_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    (tmp_path / "prepared").mkdir()

    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        load_prepared_data()
