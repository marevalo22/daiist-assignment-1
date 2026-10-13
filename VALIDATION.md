# Packaging and validation

Checked on 13 October 2026 with Python 3.12.14 and the supplied `uv.lock`.

## Completed checks

- `uv sync --locked` installed the supplied environment successfully.
- `uv run python main.py train` completed end to end, including all three model
  implementations, saved artifacts and all seven report figures.
- The notebook dashboard launched through `uv run python main.py app` and
  responded on port 7860 in 5.29 seconds, within the workflow's 30-second window.
- All seven dashboard tabs were present in the Gradio app configuration.
- Sixteen API interaction checks passed: calibration bin changes, all three
  model choices, threshold endpoints, zero costs, changed costs, feature
  selection, seating plans, event selection, person selection and pair lookup.
- Every saved artifact had the same SHA-256 before and after app testing.
- The notebook passed notebook-format validation, and the Python source passed
  syntax checks. All report image references resolved.
- Exactly one official entrypoint exists for each stage: `train.py` and
  `app.ipynb`.

The dashboard was checked through its startup response and Gradio API callbacks.
A browser visual review was not performed. Test-launch proxy settings specific
to this environment were cleared; no such changes are needed in project code.
The GitHub Actions workflow has not been run on your fork.

## Changes made while packaging

- Combined the assignment starter with the uploaded project update.
- Included the raw CSV, report images and freshly generated model artifacts.
- Restored `speeddating/evaluation.py` from function definitions in the supplied
  walkthrough notebook, and added a clear error if seating optimisation fails.
- Added `speeddating/__init__.py`.
- Converted the existing dashboard script to a sectioned `app.ipynb`, with no
  training or notebook-only installation commands.
- Made the pair explorer refresh after its event or woman selection changes,
  including when the selected man remains the same.
- Set PyTorch training to one CPU thread to reduce overhead for these small
  full-batch matrices. The model, objective and training choices are unchanged.
- Added `START_HERE.md` and a link from the assignment README.

The original combined Colab walkthrough is included as a supplementary notebook.
