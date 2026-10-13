# Run and submit this project

The official training entrypoint is `train.py`. The dashboard entrypoint is
`app.ipynb`, a notebook containing only model loading, plots and dashboard code.
`main.py` launches both. Keep exactly one format per entrypoint.

## Run it

Extract this ZIP, then open a terminal in the extracted project folder.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.
The supplied `pyproject.toml` and `uv.lock` reproduce the environment.

```bash
uv sync
uv run python main.py train
uv run python main.py app
```

Training reads `data/Speed Dating Data.csv`, saves the three models and their
preprocessor in `artifacts/`, and generates the images in `figures/`.
The dashboard loads those saved files; it never retrains anything.
Open the local URL printed by Gradio, usually http://127.0.0.1:7860.
Press Ctrl+C in the terminal to stop the app.

If saved artifacts are included, you can launch the dashboard immediately after
`uv sync`. Still run training before submission to confirm the complete pipeline
works on your machine.

On an older Intel Mac where PyTorch cannot install, the assignment recommends
GitHub Codespaces on your fork.

## Edit the dashboard as a notebook

Open `app.ipynb` in VS Code or Jupyter, select the Python interpreter from this
project's `.venv`, and run the cells in order. Start the notebook from the
repository folder so imports can find `speeddating/` and its saved artifacts.
In VS Code, use the Python and Jupyter extensions, then **Select Kernel** and
choose `.venv/bin/python` (Windows: `.venv/Scripts/python.exe`).

The notebook sections cover model loading, the seven dashboard tabs, and launch.
The official command `uv run python main.py app` converts this notebook to a
temporary Python script and executes it. Avoid notebook-only commands such as
`%pip` or code that depends on running cells out of order.

`assignment1_matias_arevalo.ipynb` is the original supplementary Colab walkthrough.
It performs both training and dashboard work. It is not an official entrypoint.

## What the files do

| File or folder | Purpose |
| --- | --- |
| `main.py` | Required launcher, retained from the assignment starter |
| `train.py` | Data checks, features, split, training, evaluation and saving |
| `app.ipynb` | Interactive Gradio dashboard, using saved models only |
| `speeddating/data.py` | Load the CSV and build one row per woman-man pair |
| `speeddating/features.py` | Build the 40 pre-event model features |
| `speeddating/torch_models.py` | Manual and standard PyTorch logistic regression |
| `speeddating/evaluation.py` | Metrics, cost rules, calibration and seating plans |
| `data/` | The committed raw dataset |
| `artifacts/` | Saved models, preprocessing and dashboard results |
| `figures/` | Images referenced by `REPORT.md` |
| `REPORT.md` | Your report; edit this before submission |
| `.github/workflows/verify.yml` | Runs training and checks app startup on GitHub |

## Put it on GitHub

A fork is your GitHub copy of the instructor's repository. A clone is your local
copy. A commit records a version, a push uploads it, and a pull request presents
your changes to the instructor.

1. Fork https://github.com/ami232/daiist-assignment-1 if you have not already.
2. Clone your fork. Replace `YOUR-USERNAME` below with your actual GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/daiist-assignment-1.git
cd daiist-assignment-1
git switch -c assignment-1
```

3. Copy the contents of this extracted project folder into the cloned repository,
   replacing files with the same names. Keep the clone's `.git` folder.
   Include `.github`, `.gitignore` and `.python-version` when copying.
4. Remove the starter's `train.ipynb` and `app.py` if they remain. This project
   uses `train.py` and `app.ipynb`; having both formats makes `main.py` stop.
5. Run the three commands in the **Run it** section and check the dashboard.
6. Review `git status`, then save and upload the changes:

```bash
git status
git add .
git commit -m "Complete speed dating assignment"
git push -u origin assignment-1
```

7. Open a pull request on GitHub, with `ami232/daiist-assignment-1` as the base
   repository and your fork's `assignment-1` branch as the source.
8. Confirm **Verify Assignment 1** passes under Actions.
9. Submit the PR link and a ZIP of the same final project on Blackboard.

Downloading or extracting a ZIP does not create a Git repository. Copy this
project into your cloned fork before using the Git commands.

## Finish the report

The report wording is preserved from the supplied update. Fill in your student
ID, email and actual group. Revise the AI disclosure to describe what you and
each tool actually contributed, consistent with the assignment's policy.
Use your own words for the findings and conclusions, and be able to explain the
pipeline, features, L2 objective, split and cost threshold.

After any final edits, rebuild the Blackboard ZIP so it matches your GitHub
submission. Include `data/`, `figures/`, `artifacts/`, source code, the notebook,
report, environment files and GitHub workflow. Exclude `.venv`, `.git`, caches
and notebook checkpoints.
