# smol-smoltalk fit within the Ventris context

Research date: 2026-10-25. Resolves the evidence question in [Research smol-smoltalk fit within the Ventris context](https://github.com/pupeno/ventris/issues/38). It does not choose the final data, prompt-format, or evaluation contract.

## Finding

**There is ample length-compatible single-turn data for the proposed 20,000-example experiment without extending the 1,024-token context or extracting turns from longer conversations.** In the first training shard alone, 43,799 complete naturally single-turn examples fit the candidate format; the full upstream test split supplies another 9,268. Even excluding the explicitly coding-oriented `self-oss-instruct` source leaves 31,951 fitting training rows in that shard.

These are structural and length checks, not a claim that 20,000 answers have been quality-vetted. A small inspection found instruction-compliance failures and source labels that do not reliably describe the task. Source balance, quality rules, duplicate handling, and the exact format still need a human decision. Source: [audit counts](instruct-data-fit-counts.json) produced by the [reproducible audit](audit_instruct_data_fit.py) against the pinned data below.

## Data and tokenizer identity

- Dataset: `HuggingFaceTB/smol-smoltalk`.
- Revision: `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`.
- Upstream splits: 460,341 training examples and 24,229 test examples. The dataset has `messages` (role/content pairs) and `source`. Its authors adapted it for models below 1B parameters, shortening conversations and removing advanced-math data. Source: [pinned dataset card](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc/README.md).
- Inspected data: all 115,086 rows in `data/train-00000-of-00004.parquet`, and all 24,229 rows in `data/test-00000-of-00001.parquet`. The other three training shards were not inspected. This is a lower-bound supply audit, not a random sample or full-training-set census.
- Local tokenizer artifact: `/workspaces/ventris/data/tokenizer.json`; SHA-256 `93b16d61c86d7905c98cfc5e30b7bf2f751d8f369a861cedb5fa1cf6f9faa422`.
- Tokenizer vocabulary: 50,257 entries; EOS `<|endoftext|>` has ID 50,256. The accompanying local config declares a 1,024-token maximum, no BOS or PAD token, and no chat template.
- Tokenization uses `tokenizers==0.23.3`, with truncation and padding disabled and `add_special_tokens=False`.

The tokenizer was taken from the current working directory. The eventual source Base checkpoints have not yet been selected and their archived tokenizers have **not** been compared with this hash. Rerun the audit against the tokenizer attached to each selected checkpoint if it differs. This limitation is independent of the dataset supply finding.

The audited Parquet file SHA-256 values are recorded in the counts artifact. Source files: [first training shard](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc/data/train-00000-of-00004.parquet), [test split](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc/data/test-00000-of-00001.parquet).

## Eligibility and candidate rendering

An example is naturally single-turn only when its entire role sequence is either `user, assistant` or `system, user, assistant`. Every message must have nonempty content after a whitespace check. Original content is retained unchanged; the optional system message is never discarded. Longer conversations are excluded as whole records. No first-turn extraction was performed, so it contributes zero examples to these counts.

The candidate rendering is ordinary text, without new special tokens:

```text
### System:
{system content}

### Instruction:
{user content}

### Response:
{assistant content}
```

When there is no system message, omit that section and its following separator. Encode the full rendered string, then count one additional EOS ID following the complete assistant answer. All headings, separators, system instructions, input text, and output text count toward the 1,024-token limit. This deliberately conservative rule counts the complete sequence including EOS; the future training contract will define its shifted input/label treatment.

This is a measuring format, not a selected template. Tokenization across message boundaries and a future response-mask boundary must be specified during implementation planning. Counts at 960 and 896 tokens show available headroom for a modest format change; they do not guarantee that every future format fits.

## Measured counts

| Measure | First training shard | Full test split |
| --- | ---: | ---: |
| All inspected rows | 115,086 | 24,229 |
| Naturally single-turn, nonempty | 45,971 | 9,733 |
| Fits at 768 tokens including EOS | 42,476 | 8,981 |
| Fits at 896 tokens including EOS | 43,271 | 9,154 |
| Fits at 960 tokens including EOS | 43,560 | 9,219 |
| Fits at 1,024 tokens including EOS | 43,799 | 9,268 |
| Distinct exact prompts among fitting rows | 43,428 | 9,219 |
| Distinct exact full records among fitting rows | 43,778 | 9,267 |

About 95.3% of the naturally single-turn training records fit. Their total formatted lengths have median 305, 75th percentile 452, 90th percentile 666, and 95th percentile 988 tokens. These quantiles describe all naturally single-turn rows in the inspected shard, including those too long to use. The maximum is 31,322, so filtering is still necessary. Source: [audit counts](instruct-data-fit-counts.json).

| Source label, preserved as published | Fitting training rows | Fitting test rows |
| --- | ---: | ---: |
| `self-oss-instruct` | 11,848 | 2,574 |
| `openhermes-50k` | 11,460 | 2,389 |
| `smol-contraints` | 8,626 | 1,799 |
| `smollm-rewrite-30k` | 6,768 | 1,418 |
| `smol-summarize-20k` | 4,066 | 856 |
| `explore-instruct-rewrite` | 794 | 183 |
| `smol-summarize-5k` | 237 | 49 |
| `smol-magpie-ultra-short` | 0 | 0 |
| `everyday-conversations` | 0 | 0 |
| `longalign` | 0 | 0 |

The dominant `smol-magpie-ultra-short` component contributes 67,651 rows to the inspected training shard, but none passes this strict natural-single-turn rule. Thus filtering changes the mixture substantially; it is not a smaller representative copy of smol-smoltalk. `self-oss-instruct` alone becomes about 27% of the fitting training pool. Removing that source is not equivalent to removing all coding tasks: the inspected OpenHermes examples also included code. Source: [source and role counts](instruct-data-fit-counts.json).

## Reproducible inspection and quality limits

For each eligible source, inspect the two fitting training records with the lexicographically smallest SHA-256 hashes of their full message JSON. This deterministically selects 14 examples across seven sources; row indices and hashes are retained in the counts artifact. These are illustrations, not a statistically sufficient quality estimate. Record indices are zero-based within the pinned first training shard.

| Source | Inspected row indices | Observation |
| --- | --- | --- |
| `smol-contraints` | 91,241; 3,272 | Formatting-focused requests; one weather answer invents a forecast despite having no location or observations. Structural compliance does not establish factual grounding. |
| `self-oss-instruct` | 77,237; 65,592 | Complete Python coding tasks, including combinatorial dictionary construction and matrix multiplication. Such tasks may exceed the intended first educational demonstration. Code was inspected, not executed or validated. |
| `smollm-rewrite-30k` | 6,277; 109,218 | Complete source passages and rewrites; the actual rewriting instruction is in the system message, which must be preserved. |
| `openhermes-50k` | 68,989; 67,230 | A copywriting/job-description transformation and a technical explanatory answer. This source is broad rather than a clean task category. |
| `smol-summarize-20k` | 14,066; 62,445 | Both include input passages and full summaries, but their answers exceed a system instruction limiting output to three sentences. The first also uses pronouns despite an instruction forbidding them. |
| `explore-instruct-rewrite` | 62,008; 6,505 | Short editing tasks fit easily; the proposed rewrite at row 6,505 still repeats the phrase it was meant to reduce. |
| `smol-summarize-5k` | 49,814; 51,259 | The samples are hobby advice and persona-styled instructions rather than summarization, demonstrating that source labels are not reliable task annotations. |

No audited message was cut to make it fit. This establishes preservation of the supplied record, not that the original authors supplied all necessary context or a correct answer. A later selection policy should review a sample of the actual selected training subset and should not assume these source names or length checks certify quality. Source: pinned records identified by [sample hashes and indices](instruct-data-fit-counts.json).

## Splits and duplication

Preserve the upstream train/test distinction while making the eventual train/validation/test policy explicit. The original splits are not free of duplicate prompts:

- Within the fitting training pool, there are 371 repeated exact-prompt occurrences and 21 repeated exact-record occurrences.
- Within the fitting test pool, there are 49 repeated exact-prompt occurrences and one repeated exact-record occurrence.
- Between the inspected training pool and full fitting test pool, there are **87 distinct overlapping exact prompts**, including **five distinct overlapping exact full records**.

An exact prompt here hashes the complete system/user message list, including role names, whitespace, and content. It does not normalize punctuation, compare the user text across differing system messages, detect paraphrases, or find overlap with external benchmarks or Base pretraining. There are at least 43,341 distinct fitting training prompts after excluding the 87 shared prompts. This is still comfortably above the training target, but duplicate policy must be applied to the actual final selection. No claim is made about the uninspected training shards. Source: [audit counts](instruct-data-fit-counts.json).

## Reproduction

Run from the research branch with the repo's dependencies installed. The cache is disposable and downloads only the first training shard plus the test split. The output contains aggregate results and the small full-text inspection sample; the checked-in counts artifact omits message text.

```bash
python docs/research/audit_instruct_data_fit.py \
  --tokenizer /path/to/ventris/data/tokenizer.json \
  --cache /tmp/ventris-smoltalk-parquet \
  --output /tmp/ventris-smoltalk-audit.json \
  --train-limit 115086
```

To regenerate the published counts artifact from the audit output, remove the `messages` key from each entry under `splits.<split>.samples.<source>` and retain the remaining JSON. No dataset contents or model weights are committed.

## Consequences for the next decision

The 1,024-token context is not a supply blocker for this pilot. The unresolved questions are the desired source/task balance, how to handle flawed instruction/answer pairs, the final prompt format, the selected checkpoint tokenizer identity, and held-out/duplicate rules. Those belong to [Choose the single-turn format and dataset preparation contract](https://github.com/pupeno/ventris/issues/39), informed by the experiment's acceptance criteria. This research does not promise that the resulting Instruct models will reach a particular quality level.
