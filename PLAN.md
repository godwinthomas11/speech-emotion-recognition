# Project Plan: Speaker-Independent Speech Emotion Recognition

**Course:** Pattern Recognition (MOOC-based), CCA1 (35 marks) → continues into CCA2 (35 marks)
**Group 19:** Aryan Gavali (1032240154) · Godwin Thomas (1032240162) · Sarthak Vyavahare (1032240163) · Praveet Gupta (1032240179)
**Plan created:** 2 November 2026
**Project folder:** `C:\Users\godwi\speech-emotion-recognition`

---

## 1. Status board

Update this table after every step. A step only moves to ✅ after its quality gate passes.

| # | Step | Status | Notes |
|---|------|--------|-------|
| 0 | Plan and runbook (this file) | ✅ Done | |
| 1 | Fill row 19 of the shared sheet | ✅ Done | Filled by Godwin on 2 Nov |
| 2 | Environment and repo skeleton | ✅ Done | `.venv` (Python 3.11), pinned `requirements.txt`, `src/ser` package, git initialised (nothing committed yet) |
| 3 | Download dataset and run EDA | ✅ Done | 1,440 clips verified; senior review passed (no leakage) | Zenodo zip verified: 208,468,073 bytes, MD5 `bc696df6…ef8a`, licence CC BY-NC-SA 4.0 |
| 4 | Feature extraction → `features.csv` | ✅ Done | 190 features, 2.5 MB, pitch fallback for 13 clips |
| 5 | Evaluation protocol and baselines | ✅ Done | Nested 6×3 actor-wise CV, 1-SE rule for k |
| 6 | Solution 1: Mutual Information + SVM | ✅ Done | F1 0.522, 102 features |
| 7 | Solution 2: RFE + Random Forest | ✅ Done | F1 0.473, 67 features |
| 8 | Solution 3: Genetic Algorithm + MLP | ✅ Done | F1 0.511, 108 features |
| 9 | Comparison, statistics and figures | ✅ Done | McNemar+Holm, ablation, leakage, speaker-norm |
| 10 | Quality review and clean re-run | ✅ Done | Two senior-review passes; all numbers verified |
| 11 | GitHub push and CCA1 form submission | ✅ Pushed | Godwin submits the forms |
| 12 | CCA2: paper support and graphical abstract | ✅ Draft ready | Group rewrites in own words, runs checks |

---

## 2. What the university asks for (and where each item is covered)

| Requirement | Where we meet it |
|---|---|
| One real-world pattern recognition case study | Speech emotion recognition (call centres, mental-health monitoring, voice assistants) |
| Three different solutions | S1, S2 and S3 below, each with its own feature selection family *and* its own classifier |
| All three implemented by the group | Notebooks 03–05, run end to end on our machine |
| **Feature selection is compulsory** (teacher's extra instruction) | Filter (MI), Wrapper (RFE) and Metaheuristic (GA): one per solution, plus a no-selection baseline to show what selection adds |
| Graphs, performance comparisons, visualisations | Notebook 06 and `figures/` (list in §7) |
| CCA1 form: code file, dataset file, GitHub link | §8 |
| CCA2: paper with all 3 solutions, graphical abstract, plagiarism < 15 %, AI content < 35 %, submitted to a venue | §10 |

---

## 3. Problem statement

Given a short spoken sentence, predict which of eight emotions the speaker is expressing: **neutral, calm, happy, sad, angry, fearful, disgust, surprised**. It must work for **speakers the model has never heard before**.

**Why "speaker-independent" matters.** Many student projects and blog posts report 85–95 % accuracy on RAVDESS. They usually split clips randomly, so the same actor's voice appears in both training and testing, and the model partly learns *who* is speaking instead of *how they feel*. We hold out whole actors in every test fold. Our accuracy will be lower (realistically around 45–65 % for 8 classes with hand-crafted features; chance level is 12.5 %), but it is honest and closer to real use. We will also **show the inflation directly** with a random-split vs actor-split comparison figure. Few groups will have this.

---

## 4. Dataset

**RAVDESS:** Ryerson Audio-Visual Database of Emotional Speech and Song.
- Citation: Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391
- Download: Zenodo record 1188976, file `Audio_Speech_Actors_01-24.zip` (~200 MB)
- License: CC BY-NC-SA 4.0 (non-commercial use with attribution, which fits academic work)
- Content (speech, audio-only): **1,440 WAV files** = 24 actors × 60 clips. 12 actors are male (odd IDs) and 12 are female (even IDs). Two fixed sentences ("Kids are talking by the door", "Dogs are sitting by the door").
- Class balance: neutral has 96 clips (no "strong" intensity); each of the other 7 emotions has 192.
- Filename code: `03-01-EE-II-SS-RR-AA.wav`, where EE is emotion, II intensity, SS statement, RR repetition and AA actor.

*All of the above is checked again against the real files in Step 3 (count, durations, sample rate, labels).*

**What we submit as the "Dataset file":** our extracted `features.csv` (1,440 rows; target under 10 MB), plus a README section that links to the original audio on Zenodo with the citation. We do not re-upload the 200 MB audio.

---

## 5. Method design

### 5.1 Pre-processing
1. Load at 22,050 Hz, mono.
2. Trim leading and trailing silence (librosa `trim`, top_db ≈ 30). RAVDESS clips have long silent edges that otherwise pollute the statistics.
3. Peak-normalise amplitude, so gain differences between recording sessions don't dominate. (As a result, RMS features describe energy dynamics within a clip, not absolute loudness. This is noted as a limitation.)

### 5.2 Feature extraction (the pattern recognition core)
Frame-level descriptors (≈ 93 ms window, ≈ 23 ms hop) are summarised over each clip with statistics (mean and std; max for RMS; percentiles for pitch). This gives a fixed-length vector of **190 features** per clip.

| Group | Descriptors | Why it relates to emotion |
|---|---|---|
| Cepstral | 20 MFCC + Δ + ΔΔ | Vocal tract shape and timbre |
| Spectral shape | centroid, bandwidth, roll-off, flatness, contrast (7 bands) | Brightness and harshness (e.g. anger is brighter) |
| Harmonic | chroma (12), tonnetz (6) | Tonal content |
| Energy | RMS energy, zero-crossing rate | Arousal: loud/excited vs calm/sad |
| Prosody | pitch (F0 via pYIN): mean, std, range, voiced fraction | Intonation (fear and surprise rise in pitch) |

The full feature list is in `docs/FEATURES.md`. For 13 clips where pYIN finds no voiced frames, pitch falls back to YIN on the louder half of the frames.

### 5.3 Evaluation protocol (applies to every model)
- **Outer loop:** 6-fold actor-wise cross-validation. Each fold tests on 4 unseen actors (2 male + 2 female), so every actor is tested exactly once.
- **Inner loop:** hyper-parameters and the number of selected features are tuned inside the training actors only (GroupKFold).
- **No leakage:** the scaler and the feature selector are re-fitted **inside every training fold** (inner and outer), never on test actors. Selecting features on the full data first is a common mistake that silently inflates results, and we avoid it.
- **Seeds:** fixed (`RANDOM_STATE = 42`) everywhere, so re-runs give the same numbers.
- **Metrics:** accuracy, macro-F1, UAR (unweighted average recall, the standard metric in emotion recognition), per-class F1, confusion matrix, number of features kept, training time and prediction time.
- **Statistics:** mean ± std across folds, and McNemar's test on pooled out-of-fold predictions to check whether differences between solutions are significant.

### 5.4 The three solutions

| | Solution 1 | Solution 2 | Solution 3 |
|---|---|---|---|
| Feature selection family | **Filter** | **Wrapper** | **Metaheuristic (evolutionary)** |
| Method | Mutual Information ranking (`mutual_info_classif`, keep top k) | Recursive Feature Elimination using RF importances | Genetic Algorithm over binary feature masks (our own NumPy implementation, so we can explain every line) |
| Classifier | SVM, RBF kernel | Random Forest | MLP neural network |
| Tuned (inner CV) | k, C, γ | k, min_samples_leaf, max_features | MLP hidden size, α (GA settings fixed: pop 20, 20 generations, patience 6) |
| Strength | Fast, model-independent | Takes feature interactions into account | Searches combinations that ranking methods miss |
| Cost | Low | Medium | High (GA is the slowest step; timing is benchmarked first) |

**GA design:** chromosome = 0/1 mask over features; fitness = MLP macro-F1 over actor-wise inner folds − 0.02 × fraction of features used; tournament (3) selection, uniform crossover (0.9), bit-flip mutation (1.5/190), elitism (2). The fitness uses the target MLP itself, so the method is a true wrapper.

### 5.5 Baselines and ablation (to prove selection helps)
Each classifier is also run on **all features with no selection**: SVM-all, RF-all, MLP-all. Every solution is compared against its own baseline, so the effect of feature selection is isolated from the effect of the classifier. A 2 × 2 swap (MI + RF, RFE + SVM) further separates selector and classifier effects. McNemar p-values are Holm-corrected, and selection stability is reported as the mean pairwise Jaccard index across folds.

---

## 6. Repository layout

```
speech-emotion-recognition/
├── README.md                 # overview, results table, how to run, citation
├── PLAN.md                   # this file
├── requirements.txt          # pinned versions
├── LICENSE                   # MIT for our code (the dataset keeps its own licence)
├── .gitignore                # excludes raw audio and the virtual env
├── data/
│   ├── raw/                  # RAVDESS WAVs (gitignored)
│   └── features.csv          # extracted features (submitted as the dataset file)
├── src/ser/                  # shared, tested code used by every notebook
│   ├── config.py             # paths, seeds, emotion labels
│   ├── dataset.py            # filename parsing, metadata table
│   ├── features.py           # feature extraction
│   ├── evaluation.py         # actor-wise CV, metrics, McNemar
│   ├── ga_selector.py        # Genetic Algorithm feature selector
│   └── plots.py              # consistent figure style
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_solution1_mi_svm.ipynb
│   ├── 04_solution2_rfe_rf.ipynb
│   ├── 05_solution3_ga_mlp.ipynb
│   └── 06_comparison.ipynb
├── results/                  # CSV/JSON metrics per fold, out-of-fold predictions
├── figures/                  # all PNG figures (300 dpi, for the paper too)
└── docs/
    ├── FEATURES.md           # every feature explained
    └── VIVA_NOTES.md         # likely questions and answers for each member
```

---

## 7. Figures we will produce

**EDA (Step 3)**
1. Class distribution (shows the neutral imbalance)
2. Example waveform for each emotion, same actor and sentence
3. Mel-spectrogram grid, one per emotion
4. Clip duration distribution before and after silence trimming

**Features (Step 4)**
5. Pitch and energy box plots per emotion
6. 2-D t-SNE / PCA projection coloured by emotion
7. Feature correlation heat map (motivates why selection is needed)

**Per solution (Steps 6–8)**
8. Accuracy vs number of selected features curve (MI and RFE)
9. GA convergence plot (best and mean fitness per generation)
10. Normalised confusion matrix for each solution
11. Top selected features for each method

**Comparison (Step 9)**
12. Grouped bar chart: accuracy / macro-F1 / UAR, each solution vs its baseline, with error bars
13. Per-class F1 heat map (solutions × emotions)
14. Selected-feature overlap between the three methods (Venn / UpSet)
15. Feature-group share chosen by each method (MFCC vs pitch vs energy …)
16. Accuracy vs training time trade-off scatter
17. **Leakage demo:** random split vs speaker-independent split accuracy

---

## 8. CCA1 submission checklist

| Form field | What we upload | Notes |
|---|---|---|
| Code file (1 file, ≤ 10 MB) | `Group19_SER_Code.zip` (src + notebooks with outputs + README) | If the form rejects ZIP, upload a single PDF of all executed notebooks instead. **Godwin: check which file types the field accepts.** |
| Dataset file | `features.csv` (+ README link to RAVDESS) | Under 10 MB; if CSV is rejected, use `.xlsx` |
| GitHub repository link | Public repo, e.g. `github.com/<username>/speech-emotion-recognition` | Godwin creates the repo; `gh` CLI isn't installed, so we push over HTTPS |

Before submitting, each member should re-run the notebooks once and read `docs/VIVA_NOTES.md`.

---

## 9. Shared sheet, row 19 (Step 1)

- **Column D (title):** Speaker-Independent Speech Emotion Recognition Using Acoustic Feature Selection and Machine Learning
- **Column E (solutions):** use Ctrl + Enter between lines
  - Solution 1: Mutual Information (Filter) Feature Selection + Support Vector Machine (SVM) – RBF Kernel
  - Solution 2: Recursive Feature Elimination (Wrapper) + Random Forest Classifier
  - Solution 3: Genetic Algorithm (Metaheuristic) Feature Selection + Multilayer Perceptron (MLP) Neural Network
- **Column F (feature selection method used):** Mutual Information (Filter), Recursive Feature Elimination (Wrapper), Genetic Algorithm (Metaheuristic)
- Add Sarthak's PRN in column B.

---

## 10. CCA2 preview (after CCA1 is submitted)

- IEEE conference format paper: Abstract, Introduction, Related Work, Dataset, Methodology (pre-processing, features, 3 solutions), Experimental Setup, Results, Discussion, Conclusion, References.
- Graphical abstract: one-image pipeline (audio → features → 3 selectors → classifiers → results).
- **Writing rule:** the group writes the paper in its own words from our results, figures and notes. That is the reliable way to stay under 15 % plagiarism and 35 % AI content. Claude can give structure, check facts and citations, and review drafts, but should not ghost-write the final text.
- Venue shortlist (conference or journal accepting student work) is prepared in Step 12.

---

## 11. Quality gates (each one must pass before moving on)

| Step | Gate |
|---|---|
| 3 | 1,440 files found; label counts match §4; no corrupt audio |
| 4 | No NaN/inf values; CSV < 10 MB; each row traceable to its filename and actor |
| 5 | CV folds contain no shared actors (asserted in code); baseline clearly above chance |
| 6–8 | Each solution compared with its own baseline; results reproducible across two runs |
| 9 | Every figure readable, labelled, 300 dpi; numbers in figures match the results CSVs |
| 10 | Fresh virtual env → all notebooks run top to bottom without errors |
| 11 | Repo public, README results table filled, files under the form's size limits |

---

## 12. Risks and how we handle them

| Risk | Plan |
|---|---|
| GA too slow on a 4-core laptop | Benchmark first; cache features; parallelise; use a proxy fitness if needed (documented) |
| Feature selection doesn't beat the baseline for some classifier | Report it honestly and explain why. Fewer features at similar accuracy is still a valid finding (faster, simpler model) |
| Accuracy looks "low" compared with online claims | The leakage demo figure explains the gap; this is a strength in the viva |
| Form rejects ZIP/CSV | PDF / XLSX fallbacks (§8) |
| Team members can't explain the work | `VIVA_NOTES.md` plus each member re-runs the notebooks |
