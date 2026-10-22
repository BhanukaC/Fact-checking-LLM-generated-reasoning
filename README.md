# Fact-checking LLM-generated reasoning

How much does evidence retrieval limit an automated verifier that checks the factual claims in an LLM's fact-checking reasoning?

Bhanuka Uyanage and Yoke Cheng Lee
Department of Computer and Systems Sciences (DSV), Stockholm University
Course project, RTDS, autumn 2026

## What this project does

Three LLMs are asked to fact-check real claims from the AVeriTeC dataset using only their own knowledge (closed-book). Each answer contains a verdict (Supported or Refuted) and a short reasoning. We split the reasoning into single factual statements and check each statement against evidence with an NLI model (entailment, neutral, contradiction). An output is flagged as possibly wrong when one of its statements is contradicted by the evidence, or when the evidence points to the opposite verdict.

AVeriTeC gives the correct verdict for every claim, so we know which LLM outputs are wrong and can measure how well the flags find them. We run the same verifier with four kinds of evidence:

| Condition | Evidence per statement |
|---|---|
| gold | the AVeriTeC gold question-answer evidence written by annotators |
| bm25 | top 5 sentences from the retrieved pool, ranked by BM25 |
| similarity | top 5 sentences ranked by embedding similarity |
| verifier | similarity top 20, reranked by the NLI model to the top 5 |

The difference between gold and the three retrieval conditions is the cost of retrieval.

Research questions:

- RQ: How much does evidence retrieval limit the reliability of an automated verifier at detecting factual errors in LLM-generated fact-checking reasoning?
- SQ1: How well does the verifier detect errors when it has the gold evidence?
- SQ2: How much does detection drop with retrieved evidence, and does the selection method (BM25, similarity, verifier-guided) matter?
- SQ3: Why does it fail? Analysis of missed errors and false flags.

## Setup

- Data: AVeriTeC dev split, Supported and Refuted claims only. 100 evaluation claims (50/50) and 20 tuning claims (10/10), sampled with seed 2026.
- LLMs (via OpenRouter, temperature 0, closed-book): `google/gemini-3.8-flash`, `anthropic/claude-sonnet-5`, `openai/gpt-6.1-sol`. 300 evaluation outputs, 32 of them wrong.
- Evidence pool: the AVeriTeC baseline's precomputed top-100 BM25 sentences per claim (`dev_top_k_sentences.json`), cleaned for the 120 sampled claims. Removed: 1,040 duplicates, 322 from fact-checking sources, 95 dated after the claim, 25 noise, 12 near copies of the claim and 8 from the claim's own source. 10,498 sentences kept (median 90 per claim, minimum 54).
- Statement extraction: GPT-6.1 splits each reasoning into fact, meta and hedge statements; only fact statements are verified (1,612 on eval). The claim itself is also verified once per condition.
- Verifier: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, passage as premise, statement as hypothesis.
- Retrieval models: `rank-bm25` (BM25Okapi) and `sentence-transformers/all-MiniLM-L6-v2`.
- Manual checks: the gold-label check found 5 claims with doubtful gold labels (99, 113, 196, 258, 330). They hold 12 of the 32 wrong outputs, so every result is also reported without them (285 outputs, 20 wrong).
- Threshold: 0.95, chosen on the tuning split. Because the tuning split has only 3 wrong outputs, the main results are threshold-free (AUROC and average precision).

## Results

Evaluation set: 300 outputs, 32 wrong (chance AP 0.107). 95% confidence intervals from 2,000 bootstrap resamples over claims.

| Condition | AUROC [95% CI] | AP [95% CI] | AUROC, statements only | Flagged at 0.95 | Precision | Recall |
|---|---|---|---|---|---|---|
| gold | 0.715 [0.589, 0.827] | 0.230 [0.107, 0.435] | 0.679 | 81 | 0.198 | 0.500 |
| bm25 | 0.594 [0.456, 0.730] | 0.136 [0.072, 0.255] | 0.595 | 171 | 0.135 | 0.719 |
| similarity | 0.587 [0.473, 0.699] | 0.128 [0.072, 0.235] | 0.565 | 181 | 0.133 | 0.750 |
| verifier | 0.599 [0.483, 0.719] | 0.137 [0.077, 0.235] | 0.609 | 233 | 0.124 | 0.906 |

In short: with gold evidence the verifier separates wrong from right outputs above chance. With retrieved evidence its ranking cannot be shown to beat chance, whichever selection method is used, and the three methods do not differ from each other. The gap to gold (+0.07 to +0.13 AUROC) is consistent but not significant with 32 errors. At the threshold, retrieved evidence produces two to three times more flags, mostly false, because many retrieved passages contradict a true statement (the passage repeats the misinformation, describes a different event, or is opinion or a sentence fragment). The full output is in `results_eval.txt`.

## Repository layout

| Script | What it does | Main output |
|---|---|---|
| `step1_sample.py` | downloads the AVeriTeC dev set and draws a stratified sample (seed 2026): 50 Supported and 50 Refuted for evaluation, 10 and 10 for tuning | `dev.json`, `sampled_claims.json` |
| `step2_generate.py` | asks the three LLMs for a verdict and reasoning (prompt v3, temperature 0, one retry on invalid JSON); `--split eval` or `--split tune`, `--limit N` for a test run; resumes where it stopped | `outputs_eval.jsonl`, `outputs_tune.jsonl` |
| `step2b_compare_versions.py` | compares prompt v2 (raw AVeriTeC dates such as `4-10-2020`) with v3 (readable dates such as `4 October 2020`): invalid outputs, right/wrong changes for claims with ambiguous and clear dates, and v3 reasoning that still states a day-month swapped date; needs `outputs_eval_v2.jsonl` | console |
| `step3_build_dataset.py` | joins the eval outputs with the gold labels, marks each output right or wrong, counts how many models got each claim right, and writes two sheets for manual checking: a spot check of 10 wrong and 10 right outputs (seed 2026), and a gold-label check of the claims every model got wrong plus claims 99 and 258. Filled sheets are never overwritten | `dataset_eval.csv`, `report_eval.txt`, `spot_check_eval.csv`, `gold_label_check.csv` |
| `step4_inspect_evidence.py` | downloads the AVeriTeC baseline's precomputed evidence (top-100 BM25 sentences per claim) once and prints its format and one entry | `data/dev_top_k_sentences.json` |
| `step4b_check_evidence.py` | checks the pool before cleaning: every sampled claim has evidence with matching claim text; per claim, counts duplicate sentences, sentences from fact-checking sites and sentences whose URL date is after the claim date; lists the top source domains; shows the evidence sentences closest to an LLM reasoning for three claims | `evidence_per_claim.csv` |
| `step4c_clean_evidence.py` | cleans the pool, checking each sentence in this order: noise (under 5 words or pingbacks), duplicate, fact-checking source (17 known domains or "fact-check" in the URL), the claim's own source or its fact-check article, a near word-for-word copy of the claim (word-sequence similarity of at least 0.75), and a URL date after the claim date. Every removed sentence is logged with its reason | `evidence_clean.json`, `evidence_removed.csv` |
| `step5_decompose.py` | splits each reasoning into statements | `statements_<split>.jsonl` |
| `step5b_check_statements.py` | checks the statement extraction | console |
| `step6_select_evidence.py` | picks evidence for gold, bm25 and similarity | `evidence_selected_<condition>_<split>.jsonl` |
| `step6b_select_verifier.py` | verifier-guided selection | `evidence_selected_verifier_<split>.jsonl` |
| `step7_verify.py` | runs the NLI verifier | `verdicts_<condition>_<split>.jsonl` |
| `step8_flag.py` | labels statements, flags outputs, chooses the threshold on tune | `threshold.json`, `flags_<condition>_eval.csv` |
| `step9_evaluate.py` | AUROC, AP, confidence intervals, paired differences, robustness and threshold sensitivity | `results_eval.txt` |
| `step10_sq3_sheets.py` | reading sheets for the failure analysis: missed errors and false flags, separately for gold and for the retrieval conditions | `sq3_missed_errors_gold.csv`, `sq3_false_flags_gold.csv`, `sq3_missed_errors_retrieval.csv`, `sq3_false_flags_retrieval.csv` |
| `step11_annotation_sheet.py`, `step11b_compare_annotation.py` | manual statement annotation and comparison with the verifier | `statement_annotation.csv` |
| `check_top_contradictions.py` | lists the 12 strongest contradictions on the tuning split for a manual check of whether the contradicted statement is really false (11 were not, 1 is contested) | `top_contradictions_tune.csv` |
| `walkthrough.py` | follows one output through the whole pipeline, e.g. `python walkthrough.py 36 claude` | console |
| `check_numbers.py` | recomputes the reported numbers from the files and prints OK or DIFF | console |

## How to run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install requests python-dotenv rank-bm25 sentence-transformers transformers torch sentencepiece protobuf
```

`dev.json` (the AVeriTeC dev split) is included; `python step1_sample.py` downloads it again from Hugging Face and recreates `sampled_claims.json` with the same seed. The evidence pool is not, because of its size: `python step4_inspect_evidence.py` downloads [`dev_top_k_sentences.json`](https://huggingface.co/chenxwh/AVeriTeC/blob/main/data_store/dev_top_k_sentences.json) (the AVeriTeC baseline's precomputed BM25 sentences) into `data/`, which is ignored by git. You only need it to redo step 4: the cleaned pool `evidence_clean.json` is included.

Only step 2 (generation) and step 5 (statement extraction) call an API. They need an OpenRouter key in a `.env` file, which is listed in `.gitignore` and never committed:

```
OPENROUTER_API_KEY=your-key-here
```

To regenerate the LLM outputs (this costs API credit, and the models may answer differently today):

```bash
python step1_sample.py
python step2_generate.py --split tune
python step2_generate.py --split eval
```

`step2_generate.py` appends to `outputs_<split>.jsonl` and skips claim and model pairs already done, so delete the file first for a fresh run.

All intermediate files are included (LLM outputs, statements, cleaned evidence, selected evidence, verdicts and flags), so any later step can be rerun on its own without a key. To rerun from evidence selection onwards (step 6 downloads the embedding model and step 7 the NLI model on first use):

```bash
python step6_select_evidence.py --split eval
python step6b_select_verifier.py --split eval
python step7_verify.py --split eval
python step8_flag.py --split eval
python step9_evaluate.py | tee results_eval.txt
python check_numbers.py
```

To reproduce only the results from the included verdicts, which takes seconds:

```bash
python step8_flag.py --split eval
python step9_evaluate.py
```

`threshold.json` is already fixed from the tuning split; `step8_flag.py --split tune` refuses to overwrite it. Earlier prompt versions are kept in `outputs_eval_v2.jsonl` and `outputs_tune_v2.jsonl` for `step2b_compare_versions.py`.

## Limitations

- Small sample: 100 claims and 32 wrong outputs give wide confidence intervals.
- Only Supported and Refuted claims are used, so results are not comparable with published AVeriTeC scores.
- The gold evidence was written to justify the gold verdict, so the gold condition is an upper bound and may partly reflect the label.
- A few fact-check sentences from general news sites (9 sentences, 4 claims) passed the cleaning; excluding the affected claims changes AUROC by at most 0.013.
- API models can change behind the same name, so regenerating step 2 may give different outputs.

## Data and licence

AVeriTeC was released by Schlichtkrull, Guo and Vlachos (NeurIPS 2023 Datasets and Benchmarks), with the data on [Hugging Face](https://huggingface.co/chenxwh/AVeriTeC). `dev.json` is copied unchanged from that repository and keeps its original licence (CC BY-NC 4.0). Files derived from it here (sampled claims, LLM outputs, statements, evidence selections and verdicts) are for non-commercial research use under the same terms.

If you use the data, please cite AVeriTeC:

```bibtex
@inproceedings{
  schlichtkrull2023averitec,
  title={{AV}eriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author={Michael Sejr Schlichtkrull and Zhijiang Guo and Andreas Vlachos},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=fKzSz0oyaI}
}
```
