"""Bounded research audit; this does not prepare training data or train a model."""

import argparse
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATASET = "HuggingFaceTB/smol-smoltalk"
REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=60000)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    eos = tokenizer.token_to_id("<|endoftext|>")
    assert eos is not None
    result = {
        "dataset": DATASET,
        "revision": REVISION,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos,
        "train_limit": args.train_limit,
        "splits": {},
    }
    seen_prompts = {}
    seen_records = {}
    for split, filename, limit in [
        ("train", "train-00000-of-00004.parquet", args.train_limit),
        ("test", "test-00000-of-00001.parquet", None),
    ]:
        path = args.cache / filename
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{filename}"
        if not path.exists():
            print(f"Downloading {split}: {filename}", flush=True)
            urllib.request.urlretrieve(url, path)
        print(f"Scanning {split}", flush=True)
        report, prompts, records = scan(path, tokenizer, limit)
        report["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["file"] = filename
        result["splits"][split] = report
        seen_prompts[split] = prompts
        seen_records[split] = records
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["cross_split_eligible_exact_prompt_overlap"] = len(
        seen_prompts["train"] & seen_prompts["test"]
    )
    result["cross_split_eligible_exact_record_overlap"] = len(
        seen_records["train"] & seen_records["test"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


def scan(path, tokenizer, limit):
    parquet = pq.ParquetFile(path)
    patterns = Counter()
    sources = Counter()
    natural_sources = Counter()
    eligible_sources = Counter()
    lengths = []
    prompts = set()
    records = set()
    samples = {}
    empty = 0
    scanned = 0
    duplicate_prompt = 0
    duplicate_record = 0
    started = time.monotonic()
    for batch in parquet.iter_batches(batch_size=512, columns=["messages", "source"]):
        rows = batch.to_pylist()
        if limit is not None:
            rows = rows[: max(0, limit - scanned)]
        natural = []
        for offset, row in enumerate(rows):
            messages = row["messages"]
            source = row["source"]
            roles = tuple(message["role"] for message in messages)
            patterns["/".join(roles)] += 1
            sources[source] += 1
            if roles not in [("user", "assistant"), ("system", "user", "assistant")]:
                continue
            if any(not message["content"].strip() for message in messages):
                empty += 1
                continue
            natural_sources[source] += 1
            natural.append((scanned + offset, source, messages))
        texts = [render(messages) for _, _, messages in natural]
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        for (index, source, messages), encoding in zip(natural, encoded, strict=True):
            length = len(encoding.ids) + 1  # One EOS id after the complete assistant answer.
            lengths.append(length)
            if length > 1024:
                continue
            eligible_sources[source] += 1
            prompt_hash = content_hash(messages[:-1])
            record_hash = content_hash(messages)
            duplicate_prompt += prompt_hash in prompts
            duplicate_record += record_hash in records
            prompts.add(prompt_hash)
            records.add(record_hash)
            sample = {
                "row_index": index,
                "tokens": length,
                "record_sha256": record_hash,
                "messages": messages,
            }
            pool = samples.setdefault(source, [])
            pool.append(sample)
            pool.sort(key=lambda item: item["record_sha256"])
            del pool[2:]
        scanned += len(rows)
        if scanned % 10240 == 0:
            print(
                f"  {scanned} scanned; {sum(eligible_sources.values())} eligible; "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
        if limit is not None and scanned >= limit:
            break
    ordered = sorted(lengths)
    return (
        {
            "file_rows": parquet.metadata.num_rows,
            "scanned": scanned,
            "role_patterns": dict(patterns),
            "sources": dict(sources),
            "natural_single_turn_nonempty": len(lengths),
            "natural_sources": dict(natural_sources),
            "empty_natural_records": empty,
            "eligible_at_1024": sum(eligible_sources.values()),
            "eligible_sources": dict(eligible_sources),
            "fit_counts": {str(n): sum(x <= n for x in lengths) for n in [768, 896, 960, 1024]},
            "natural_length_quantiles": {
                str(q): ordered[round((len(ordered) - 1) * q)]
                for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
            },
            "eligible_duplicate_exact_prompts": duplicate_prompt,
            "eligible_duplicate_exact_records": duplicate_record,
            "eligible_unique_exact_prompts": len(prompts),
            "eligible_unique_exact_records": len(records),
            "samples": samples,
        },
        prompts,
        records,
    )


def render(messages):
    headings = {"system": "System", "user": "Instruction", "assistant": "Response"}
    return "\n\n".join(
        f"### {headings[message['role']]}:\n{message['content']}" for message in messages
    )


def content_hash(messages):
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


if __name__ == "__main__":
    main()
