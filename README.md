# Speaker-Independent Speech Emotion Recognition Using Acoustic Feature Selection

Pattern Recognition (CCA1) case study, Group 19:
Aryan Gavali · Godwin Thomas · Sarthak Vyavahare · Praveet Gupta

We classify short spoken sentences into eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) using the RAVDESS dataset. Every model is tested on **actors it never heard during training**, and every solution uses a **feature selection algorithm**:

| Solution | Feature selection | Classifier |
|---|---|---|
| 1 | Mutual Information (filter) | SVM, RBF kernel |
| 2 | Recursive Feature Elimination (wrapper) | Random Forest |
| 3 | Genetic Algorithm (metaheuristic) | Multilayer Perceptron |

## Results

*Filled in after the experiments are run (Step 9).*

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

Download `Audio_Speech_Actors_01-24.zip` from [Zenodo record 1188976](https://zenodo.org/records/1188976) and extract it into `data/raw/`. You should get 24 folders `Actor_01` … `Actor_24` with 1,440 WAV files in total.

Then run the notebooks in `notebooks/` in order (01 → 06).

## Repository layout

```
data/features.csv   extracted acoustic features (one row per clip)
src/ser/            shared code used by all notebooks
notebooks/          one notebook per step: EDA, features, 3 solutions, comparison
results/            per-fold metrics and predictions
figures/            all plots
docs/               feature descriptions and viva notes
```

## Dataset and citation

RAVDESS is by Steven R. Livingstone and Frank A. Russo and is licensed under CC BY-NC-SA 4.0. We use only the audio-only speech portion (1,440 clips, 24 professional actors, 12 male and 12 female).

> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

Our code is under the MIT licence (see `LICENSE`).
