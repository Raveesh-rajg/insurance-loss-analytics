# Insurance Loss & Referral Analytics

Compare referral-score signal families and calculate paid loss ratios and valuation-aware development triangles.

## Implementation and validation

Local analytics and 12 regression checks pass. Future development periods remain null; factors use matched observed cells. The Power BI report in docs is a build specification, not a rendered report.

Automated checks: **12 tests**. The GitHub Actions run linked above the file browser is the current CI result. Local checks and external integrations are separate claims.

## Reproduce locally

Use Python 3.12. Run from this repository’s root in a fresh virtual environment.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

For repositories using `src/`, set the import path before running commands:

```powershell
# PowerShell
$env:PYTHONPATH="src"
```
```sh
# macOS/Linux
export PYTHONPATH=src
```

```sh
python run_pipeline.py
python -m pytest tests -q
```

## Data and interpretation

Synthetic policies, claims and planted fraud rings. Referral scores are research signals, not decisions about real people. Paid loss is not incurred loss; annual premium allocation is a simplified denominator.

## Inspect the work

- [`tests/`](tests/) — executable checks and examples.
- [`docs/`](docs/) — methodology, integration specifications and the historical design.
- [Portfolio](https://raveesh-rajg.github.io/) — project directory.

## Completion boundary

Passing local tests establishes the checks listed in this repository. It does not establish cloud deployment, real-data quality, production security, or native BI rendering unless an explicit verification record says so.

## Exported evidence

[`outputs/`](outputs/) contains regenerated CSV tables and `verification.json`. Run `python run_pipeline.py` to rebuild every export. These are analytical outputs, not screenshots of a native BI report.
