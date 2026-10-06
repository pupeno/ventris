"""Load and tokenize FineWeb-Edu for language-model training."""

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from datasets import Dataset, DatasetDict, Features, List, Value, load_dataset, load_from_disk
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CORPUS_REPOSITORY = "HuggingFaceFW/fineweb-edu"
CORPUS_CONFIGURATION = "sample-10BT"
CORPUS_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
VALIDATION_FRACTION = 0.01
DATA_SEED = 0

VOCAB_SIZE = 50_257
SEQUENCE_LENGTH = 1_024
EOS_TEXT = "<|endoftext|>"

# Heuristic guesses; these tokenizer-training sample limits have not been tuned.
TOKENIZER_TRAINING_DOCUMENTS = 500_000
TOKENIZER_DOCUMENT_CHARACTERS = 10_000
DOCUMENT_BATCH_SIZE = 256


def train_tokenizer(output_dir: Path | None = None) -> Path:
    """Train the byte-level BPE tokenizer on training documents only."""
    output_dir = output_dir or DATA_DIR
    training = _corpus_splits()["train"]
    sample_size = min(TOKENIZER_TRAINING_DOCUMENTS, len(training))
    sample = training.shuffle(seed=DATA_SEED).select(range(sample_size))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        _training_text(sample),
        trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE - 1,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
        length=len(sample),
    )
    tokenizer.add_special_tokens([AddedToken(EOS_TEXT, special=True, normalized=False)])

    output_dir.mkdir(parents=True, exist_ok=True)
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS_TEXT,
        model_max_length=SEQUENCE_LENGTH,
    ).save_pretrained(output_dir)
    return output_dir


def prepare_data() -> DatasetDict:
    """Tokenize the corpus and save packed sequences for training."""
    path = DATA_DIR / "prepared"
    ready = path / ".complete"
    ready.unlink(missing_ok=True)
    tokenizer = load_tokenizer()
    features = Features(
        {
            "input_ids": List(Value("uint16"), length=SEQUENCE_LENGTH + 1),
        }
    )
    with TemporaryDirectory(prefix=".packing-", dir=DATA_DIR) as cache_dir:
        prepared = DatasetDict(
            {
                name: Dataset.from_generator(
                    _pack_documents,
                    features=features,
                    cache_dir=cache_dir,
                    gen_kwargs={"documents": documents, "tokenizer": tokenizer},
                )
                for name, documents in _corpus_splits().items()
            }
        )
        prepared.save_to_disk(path)
        ready.write_text("ready\n")
    return load_prepared_data()


def load_prepared_data() -> DatasetDict:
    """Load only data whose preparation finished successfully."""
    path = DATA_DIR / "prepared"
    if not (path / ".complete").is_file():
        raise FileNotFoundError(
            f"prepared training data is missing or incomplete at {path}; "
            "run scripts/prepare_data.py first"
        )
    prepared = load_from_disk(path)
    assert isinstance(prepared, DatasetDict)
    return prepared.with_format("torch")


def load_tokenizer(path: Path | None = None) -> PreTrainedTokenizerFast:
    """Load the trained tokenizer."""
    path = path or DATA_DIR
    if not (path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(
            f"trained tokenizer not found at {path}; run scripts/train_tokenizer.py first"
        )
    return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)


def _corpus_splits() -> DatasetDict:
    splits = _load_corpus().train_test_split(
        test_size=VALIDATION_FRACTION,
        seed=DATA_SEED,
    )
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def _load_corpus() -> Dataset:
    corpus = load_dataset(
        CORPUS_REPOSITORY,
        CORPUS_CONFIGURATION,
        split="train",
        revision=CORPUS_REVISION,
    )
    assert isinstance(corpus, Dataset)
    return corpus.select_columns("text")


def _training_text(documents: Dataset) -> Iterator[list[str]]:
    for batch in documents.iter(batch_size=DOCUMENT_BATCH_SIZE):
        texts = cast(dict[str, list[str]], batch)["text"]
        yield [text[:TOKENIZER_DOCUMENT_CHARACTERS] for text in texts]


def _pack_documents(
    documents: Dataset,
    tokenizer: PreTrainedTokenizerFast,
) -> Iterator[dict[str, list[int]]]:
    """Yield sequences with one extra token for next-token labels.

    With sequence length 3, tokens [a, b, c, d, e, f, g] produce [a, b, c, d]
    and [d, e, f, g]: the shared token is a label first, then the next input.
    """
    remainder: list[int] = []
    eos_id = tokenizer.eos_token_id
    assert isinstance(eos_id, int)
    for batch in documents.iter(batch_size=DOCUMENT_BATCH_SIZE):
        texts = cast(dict[str, list[str]], batch)["text"]
        encodings = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for input_ids in encodings:
            remainder.extend(input_ids)
            remainder.append(eos_id)

        sequence_count = max(len(remainder) - 1, 0) // SEQUENCE_LENGTH
        for index in range(sequence_count):
            start = index * SEQUENCE_LENGTH
            sequence = remainder[start : start + SEQUENCE_LENGTH + 1]
            yield {"input_ids": sequence}
        remainder = remainder[sequence_count * SEQUENCE_LENGTH :]
