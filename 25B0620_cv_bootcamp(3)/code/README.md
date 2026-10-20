# Inter-IIT Tech Meet 15.0 — CV Bootcamp submission

Roll number: 25B0620  
Name: Afnan Ahmed

## Submission contents
- `predictions/`: required test predictions, Task 2 training dynamics, and Task 3 noise scores.
- `results.json`: recorded experiment settings and validation metrics.
- `report.md`: answers to Q1–Q5 (each under the 100-word limit).
- `task1_confusion_matrix.csv`: 10×10 confusion matrix (rows = clean labels; columns = noisy labels), computed from the supplied `labels.csv`.
- `code/kaggle_notebook_export.md`: available exported Kaggle notebook cells/output covering Tasks 1–5.
- `code/validate_outputs.py`: local lightweight format validation.

## Validate the packaged outputs
From the extracted submission root:

```bash
python code/validate_outputs.py
```

Run the official format checker from the provided bootcamp folder:

```bash
python check_format.py /path/to/25B0620_cv_bootcamp.zip
```

The official checker validates required files, CSV formats, IDs, score coverage, metadata, and report headings. It does not verify model accuracy or training provenance.

## Reproduction caveat
The supplied notebook export is a Markdown export of notebook cells and output, not a self-contained executable `.ipynb`; original checkpoints and a standalone end-to-end training script were not present among the finalization artifacts. Therefore the predictions and recorded metrics are packaged for submission, but full retraining from this ZIP alone is not guaranteed. Do not interpret the format check as validation of model quality.
