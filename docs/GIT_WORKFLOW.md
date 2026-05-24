# Git Workflow

This workflow keeps the project history readable and makes future report/code updates easier to review.

## Branches

| Branch Type | Naming Pattern | Example |
|---|---|---|
| Main branch | `main` | `main` |
| Documentation | `docs/<short-topic>` | `docs/readme-refresh` |
| Experiment code | `experiment/<short-topic>` | `experiment/zero2-config` |
| Results update | `results/<short-topic>` | `results/deepspeed-runpod` |
| Fixes | `fix/<short-topic>` | `fix/gpu-log-parser` |

## Commit Style

Use short imperative commit messages:

```text
Add project README
Document DeepSpeed run workflow
Update ZeRO-2 fair run command
Refresh presentation assets
```

For larger commits, add a short body:

```text
Update DeepSpeed run workflow

- Align helper script with the fair 1,000-sample comparison run
- Document GPU logging and telemetry parsing
- Keep result paths consistent with the final report
```

## Recommended Commit Sequence

For this project, a clean initial history can be built in this order:

| Step | Commit | Scope |
|---:|---|---|
| 1 | `Add project documentation` | Root README, DeepSpeed README, workflow docs |
| 2 | `Add DeepSpeed training implementation` | ZeRO-2 config, training script, parser, run commands |
| 3 | `Add QLoRA notebook and report assets` | Notebook, PDF report, slide generation scripts |
| 4 | `Add captured experiment results` | Metrics, logs, trainer state, generated slides |

If the project is already being committed as one initial upload, use:

```text
Initial project import
```

## Before Each Commit

Run the relevant checks for the files you changed:

| Change | Check |
|---|---|
| Python scripts | `python -m compileall DeepSpeed generate_slides.py generate_polished_slides.py` |
| DeepSpeed config | `python -m json.tool DeepSpeed/ds_zero2_config.json` |
| Results JSON | `python -m json.tool DeepSpeed/results/metrics_zero2_fair.json` |
| README/docs | Review Markdown preview on GitHub |
| Slides | Regenerate with `python generate_polished_slides.py` when chart inputs change |

## Publishing To GitHub

```bash
git status
git add README.md docs/GIT_WORKFLOW.md DeepSpeed/README.md .gitignore
git commit -m "Add project documentation"
git push -u origin main
```

If the GitHub repository does not exist yet:

```bash
gh repo create PDC_DeepSpeed_Project --public --source . --remote origin --push
```

Use `--private` instead of `--public` if the project should not be publicly visible.
