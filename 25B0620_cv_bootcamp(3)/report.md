# Inter-IIT Tech Meet 15.0 — Computer Vision Bootcamp

**Name:** Afnan Ahmed  
**Roll number:** 25B0620

## Q1 — Label Noise Analysis
The measured noisy-versus-clean mismatch rate is 20,104/50,000 = 40.208%. The largest directional confusions in the supplied labels are clean truck→noisy automobile (1,410), clean automobile→noisy truck (1,237), clean cat→noisy dog (1,028), and clean dog→noisy cat (915). These pairs are visually similar. The full 10×10 confusion matrix is included as `task1_confusion_matrix.csv` at the submission root.

## Q2 — Memorization and Early Stopping
Wrong-label memorization becomes apparent around epoch 40: training accuracy on the mislabeled subset rises from 17.41% at epoch 39 to 21.47% at epoch 40, while noisy validation accuracy drops from its peak of 55.20% at epoch 39 to 53.90%. By epoch 60, mislabeled-subset training accuracy reaches 68.48% while noisy validation accuracy is 50.56%. Epoch 39 was selected for early stopping using noisy validation accuracy.

## Q3 — Noise Scoring
The score is one minus the out-of-fold probability assigned to each image's observed noisy label; larger scores indicate a label less compatible with the model. This signal can fail when a genuinely difficult, ambiguous, or atypical image receives low confidence despite having a correct label. The score file contains all 45,000 training indices. The original fold-generation procedure is documented in the supplied notebook export, but the packaged outputs alone cannot independently prove the training provenance.

## Q4 — Robust Training Methods
Generalized cross-entropy (q=0.7) achieved the highest reported noisy-validation accuracy (57.16%), compared with 56.58% for label smoothing and 56.42% for filtered cross-entropy. GCE reduces the influence of high-loss examples, which may include mislabeled samples, without requiring every such sample to be removed. The ranking is based on the recorded noisy-validation measurements.

## Q5 — Synthetic Labels
The synthetic-label cross-entropy baseline reached 49.66% validation accuracy, while filtered GCE reached 50.70%, a 1.04-point gain. The combined method helped in this experiment. Because the artifacts do not include synthetic-label runs for every Task 4 method, a complete real-versus-synthetic ranking comparison cannot be established.

## Reproducibility note
The ZIP includes the available Kaggle notebook export and packaged outputs. Training scripts/checkpoints were not separately available as standalone executable files, so full retraining from this ZIP alone is not guaranteed. The official format checker validates file structure and CSV formats, not accuracy or training provenance.
