# Small instruction datasets for Ventris

Research date: 2026-10-25. This note informs the Instruct wayfinding discussion; it does not select a dataset or resolve a decision ticket.

## Recommendation to discuss

For the simplest small first experiment, shortlist **Dolly-15k**. For a dataset chosen specifically for models near Ventris's size, shortlist a **fixed, length-filtered sample of smol-smoltalk**, initially around 10,000–20,000 examples. That sample size is an experimental starting point, not a published optimum. Keep the book dataset as a cheap pipeline baseline. There is no primary-source comparison establishing which wins on Ventris.

| Dataset | Available examples | Structure and origin | Why consider it |
| --- | --- | --- | --- |
| Raschka book dataset | 1,100 | `instruction`, optional-content `input`, `output` | Smallest and easiest baseline; matches the book's workflow. |
| Dolly-15k | 15,011 | Human-written `instruction`, `context`, `response`, `category` | Compact, understandable single-turn data across eight categories. |
| Alpaca / cleaned Alpaca | About 52,000 | Synthetic instruction/input/output records; original answers from text-davinci-003 | Familiar baseline and simple schema; cleaned version repairs known data defects. |
| Alpaca-GPT4 | 52,000 English examples | Alpaca's prompts with GPT-4-generated responses | A simple alternative to original Alpaca with a different response generator. |
| smol-smoltalk | 460,341 train; 24,229 test | `messages` with role/content entries, plus `source`; a mixture including synthetic conversations | Explicitly adapted for sub-billion-parameter models, including 135M. Use a subset if small data is the priority. |
| LIMA | 1,000 training prompts/responses in the paper | Carefully curated examples | Interesting quality-over-quantity reference, but its headline evidence is at 65B. |

## Evidence and limitations

**Book dataset.** The linked JSON contains exactly 1,100 records. The chapter notebook uses pretrained GPT-2 medium (355M) with a 1,024-token context for its principal fine-tuning example. This is useful pedagogical precedent, but not evidence of equivalent outcomes with Ventris's different Base checkpoints. These facts were checked against the [JSON](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json) and [chapter notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb).

**Dolly-15k.** Databricks employees wrote the records and were instructed not to use generative AI. Categories cover tasks such as classification, extraction, summarization, question answering, brainstorming, and creative writing. The optional context matters: it can supply the passage the answer depends on. Human authorship is not a quality guarantee; the card acknowledges factual errors, biases, and deliberately lightweight annotation guidance. It has one train split, so we would need to establish our own held-out split. Its declared license is CC BY-SA 3.0. Sources: [official dataset card](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [exact size metadata](https://datasets-server.huggingface.co/size?dataset=databricks/databricks-dolly-15k).

**Alpaca family.** Stanford's original collection has 52K examples generated using text-davinci-003. The cleaned release documents repairs to malformed, missing, or incorrect records, but those repairs do not establish comprehensive correctness. Its displayed size is about 51.8K; use a pinned revision and count the loaded data rather than relying on the inherited README count. Alpaca-GPT4 uses the same instruction collection with GPT-4 responses and specifies research-only CC BY-NC usage. Do not assume the cleaned mirror's license metadata overrides upstream restrictions. Sources: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned), [GPT-4-LLM repository](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM).

**smol-smoltalk.** Its authors explicitly tailored it for models below 1B parameters and used it for SmolLM2-135M and 360M. Relative to full SmolTalk, it has shorter conversations, less task-specific data, and no advanced-math datasets. It includes both simple exchanges and longer message lists, so it does not force a single-turn or multi-turn product decision. Exact counts above come from the public size API. Sources: [official dataset card](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), [size metadata](https://datasets-server.huggingface.co/size?dataset=HuggingFaceTB/smol-smoltalk).

The SmolLM2-135M Base model was pretrained on 2 trillion tokens, and its released Instruct model also underwent DPO after SFT. Consequently its reported scores do not isolate the effect of this SFT dataset, and a small subset on Ventris cannot be assumed to reproduce them. Source: [official SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). The full SmolTalk card distinguishes Apache-2.0 for newly generated components from the licenses of reused datasets; inspect component provenance when selecting a subset. Source: [SmolTalk composition and licenses](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).

**LIMA.** The paper's result uses 1,000 curated examples to fine-tune a 65B LLaMA model. It supports investigating data quality, not assuming that 1,000 examples suffice for a 124M model. The official data repository is gated and states CC BY-NC-SA or stricter source terms. Sources: [paper](https://arxiv.org/abs/2305.11206), [official data](https://huggingface.co/datasets/GAIR/lima).

## Implications for the eventual spec

These are proposed experiment controls, not settled design decisions:

- Start by evaluating ordinary instruction following: brief answers, rewriting, classification, and extraction. Multi-turn conversation can remain a separate decision.
- Measure full formatted prompt-plus-response lengths using Ventris's tokenizer. The existing model defaults use a 1,024-token context. Filter or deliberately select complete examples; silently cutting off an answer or its required context changes the training task.
- Pin the dataset revision, sampling seed, filtering rules, and split. Sample across relevant categories or sources; do not take the first rows of an ordered dataset.
- Keep validation and evaluation examples outside training. Inspect a small sample manually and compare generated responses to the same Base checkpoint on held-out prompts.
- Judge the recipe by held-out instruction following as well as training loss. Dataset size alone does not establish quality, and none of the sources above provides a direct comparison on Ventris.

Repository context: [README](../../README.md), [GLOSSARY](../../GLOSSARY.md). No training, code changes, or tracker decisions were performed for this research.
