# System Prompt Design & Evaluation

Three versions of a system prompt for a support-triage assistant, evaluated on 15 test inputs x 2 runs each.
Eval model: `openai/gpt-oss-20b` (Groq). Judge: `openai/gpt-oss-120b`.

## What's inside
```
prompts/        v1.txt (baseline), v2.txt (structured), v3.txt (few-shot + edge cases)
eval_set.json   15 cases: clear, ambiguous, angry, very short, Roman Urdu, multi-issue, prompt injection
run_eval.py     harness: runs all prompts, scores outputs, writes results/
results/        outputs.json (raw), scores.csv (per-run), summary.csv/md, chart.png
CHANGELOG.md    what changed between versions, why, results and limitations
Prompt_Eval.ipynb   one-click Google Colab version
```

## Scoring (7 checks per output)
| Check | How |
|---|---|
| json_valid | code: parses as JSON with category/urgency/reply |
| category | code: matches expected category |
| urgency_valid | code: low / medium / high |
| length_ok | code: reply <= 80 words |
| injection_safe | code: output doesn't contain forbidden strings on injection cases |
| professional | LLM-as-judge (tone) |
| language_match | code: Latin script; Roman Urdu cases must use Roman Urdu words |

Also reports **consistency**: % of cases where all runs gave the same category.

## Results
| version | overall % |
|---|---|
| v1 | 54.3 |
| v2 | 92.9 |
| v3 | 100.0 |

Details, failures and limitations: see `CHANGELOG.md` and `results/summary.md`.

## Run it
```bash
pip install requests anthropic matplotlib
export GROQ_API_KEY=gsk_...          # or ANTHROPIC_API_KEY=sk-ant-...
python run_eval.py --runs 2 --model openai/gpt-oss-20b --judge-model openai/gpt-oss-120b
python run_eval.py --dry-run         # test the pipeline offline with fake outputs
python run_eval.py --rescore         # re-score saved outputs without regenerating
python run_eval.py --fill-empty --model <same model>   # regenerate empty outputs, then re-score
```
