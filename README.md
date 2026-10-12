# Selection Efficiency audit: full rerun

Regenerates every number in the paper from one cached set of generations. Generation happens once; every selector, ablation and table is then computed from the same saved files, so the tables cannot disagree with each other again.

## Pipeline

| Step | Command | Needs GPU | What it writes |
|---|---|---|---|
| 1. Data | `python data.py` | no | `outputs/data/{wol4,wol5,ma1,ma2}.jsonl` |
| 2. Generate | `python generate.py --model <key>` | yes | `outputs/gen/<model>/<split>.jsonl` (3 style traces + 5 self-consistency samples per item) and `<split>.judge.jsonl` |
| 3. Score | `python score.py` | yes (small) | NLI edge probabilities (RoBERTa + BART) and trace cosine |
| 4. Analyze | `python analyze.py` | no | `outputs/tables/*.tex`, `results.json`, `claims.md` |

Model keys: `qwen1.5b`, `qwen3b`, `phi2`, `smollm2` (SmolLM2 is the capability-floor appendix case).

Every step is resumable. If Colab disconnects, rerun the same command and it continues from the last saved item. Set `SE_OUT` to a Google Drive path so outputs survive the session (the notebook does this).

**Estimated time on a free T4:** about 10–20 min per model/split for generation, so about 3–4 h for all four models. Scoring takes about 10 min and analysis a few seconds. If you hit CUDA out-of-memory, lower `GEN_BATCH` for that model in `config.py`. Note that the batch size is part of the seeding, so keep it fixed once a run has started.

## Running it

Use `run_se_audit.ipynb` in Colab (Runtime → T4 GPU), or on any GPU machine run:

```bash
pip install -r requirements.txt
bash run_all.sh
```

## Using the output in the paper

Each `outputs/tables/tab_*.tex` is a complete `table` environment carrying the paper's existing label. To use one, replace the table in `colm2026_conference.tex` with `\input{tables/tab_methods_full}` (and so on). The preamble must define `\negv` and `\SE`; your paper already does.

`claims.md` lists every number quoted in the text next to its recomputed value. Update the abstract, §4, §5 and the conclusion from it.

## Method details the camera-ready text must match

These are fixed by this code. Some differ from or add to the submitted text.

1. **Random baseline.** `A_rand` is the *expected* accuracy of a uniformly random style trace (the mean per-item fraction correct), not a single random draw. This makes it equal to Single-$p$ in the decomposition.
2. **Ties.** Plurality ties and equal trust scores are scored in expectation over the tied candidates. This is the expected value of "break ties uniformly at random", with no seed noise.
3. **SE reference.** All selectors use the style pool's `A_rand`/`A_oracle` (shared reference). The new `tab:pool` table also gives self-consistency's own-pool SE.
4. **Data.**
   - **WoL-4** is 50 BBH `web_of_lies` items.
   - **WoL-5** is 50 items generated in the identical BBH template at chain length 5, balanced Yes/No. BBH ships only length 4.
   - **MA-1/MA-2** are MultiArith problems with additive-only (+, −) and multiplicative (×, ÷) equations. Every MultiArith problem has exactly two operations.
   - All splits use seed 0.
5. **Prompts.** The three persona prompts, the neutral self-consistency prompt and the judge prompt are in `prompts.py`. Quote them in the appendix.
6. **ToTh.** Traces are split into steps (one per line or sentence). Each adjacent pair is scored by RoBERTa-large-MNLI, and the trace with the highest trust score is picked. The six mappings V0–V5 and the inverted mapping are defined in `analyze.py::trust` and in the `tab:mappings` caption. V0 is the headline mapping.
7. **BART ablation** now covers all configurations, not five.
8. **Cosine** uses `all-MiniLM-L6-v2` sentence embeddings. ROUGE-L is LCS F-measure on whitespace tokens.
9. **Symbolic verifier.** The *solver* computes the answer from the question alone, so it is 100% by construction (verified against all 250 BBH labels). The *verifier pick* selects the style trace that agrees with it, which reaches the oracle. The submitted text's "reaches 100%" describes the solver.
10. **k-matched control** averages a plurality vote over all C(5,3)=10 subsets of the cached self-consistency samples. No extra generation is needed.

## Files

`config.py` (all settings) · `data.py` · `wol.py` (Web of Lies parser/solver/generator/verifier) · `prompts.py` (prompts, answer extraction, step splitting) · `generate.py` · `score.py` · `analyze.py` · `run_all.sh` · `run_se_audit.ipynb`

## Troubleshooting

- **CUDA out of memory:** lower `GEN_BATCH[<model>]` in `config.py` before starting that model.
- **Phi-2 produces garbage or `!!!!`:** this is fp16 overflow. Delete `outputs/gen/phi2/` and rerun with `--dtype float32`, and halve `GEN_BATCH["phi2"]`.
- **Low answer-parse rate** (`claims.md` reports the unparseable-answer rate): inspect a few traces in `gen/<model>/<split>.jsonl` before trusting the tables.

## Running on Kaggle

Import `run_se_audit_kaggle.ipynb` into Kaggle once (File → Import Notebook) and set `REPO` in its first cell. Every run clones or pulls this repository, so code fixes need no re-upload. All logic lives in `kaggle_run.py`.

1. In the notebook settings, set **Accelerator = GPU T4 x2** and **Internet = On**. T4 x2 is recommended over P100, because recent PyTorch builds may not support the P100. With a single GPU, the models run one after another.
2. Run all cells with `FULL = False`. This is a smoke test: 10 items per split on Qwen-1.5B and 6 on Phi-2, written to a separate folder.
3. Set `FULL = True` and use **Save Version → Save & Run All (Commit)**. `tables.zip` and `raw_outputs.zip` appear in the notebook's Output tab.
4. If a committed run ends incomplete, it skips scoring, lists the short files and still saves its outputs. Add that notebook's output as an input (**Add Input → Notebook Output Files**) and commit again. The run copies the earlier output back and continues where it stopped.

Interactive Kaggle sessions delete their files when they end, so run the full job as a committed version.
