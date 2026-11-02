# Viva notes: Group 19

Read this once before the viva. Each answer is short enough to say out loud. The final numbers are in `README.md` (results table) and `results/summary_table.csv`.

## Suggested split (who explains what)

| Member | Part to own | Notebook |
|---|---|---|
| Aryan Gavali | Dataset, EDA, feature extraction | 01, 02 |
| Godwin Thomas | Evaluation protocol, Solution 1 (MI + SVM), comparison | 03, 06 |
| Sarthak Vyavahare | Solution 2 (RFE + Random Forest) | 04 |
| Praveet Gupta | Solution 3 (Genetic Algorithm + MLP) | 05 |

Everyone should be able to answer the "general" questions below.

---

## General

**What problem are you solving?**
Recognising one of eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from a short spoken sentence, for speakers the system has never heard.

**Why is it a pattern recognition problem?**
It follows the classic chain: raw signal → feature extraction (190 acoustic measurements) → feature selection → classification → evaluation.

**Which dataset, and why?**
RAVDESS (Livingstone & Russo, 2018): 1,440 clips, 24 professional actors (12 male, 12 female). Both sentences are fixed, so only the emotion changes, which makes it a clean benchmark. Its licence (CC BY-NC-SA 4.0) allows academic use.

**Why is neutral smaller (96 vs 192 clips)?**
Neutral was not recorded at "strong" intensity. That's why we report macro-F1 and UAR, which give every class equal weight, and use balanced class weights.

**What does "speaker-independent" mean, and why does it matter?**
The test actors are never in the training set. In our t-SNE plot the clips split clearly by gender, so voice identity is the strongest pattern in the data. With a random split the model can partly recognise *who* is speaking. Our leakage experiment shows how much a random split inflates accuracy (see notebook 06, §7).

**Explain your cross-validation.**
Nested and grouped by actor. The outer loop has 6 folds, and each tests 4 unseen actors (2 male + 2 female). Inside the 20 training actors, a 3-fold actor-wise split chooses the number of features and the hyper-parameters. The scaler and the feature selector are refitted inside every training set, so the test actors never influence any decision.

**Why not select features once on the whole dataset first?**
That is selection bias (Ambroise & McLachlan, 2002). The test data would have influenced which features were picked, and the reported accuracy would be optimistic.

**Why macro-F1 and UAR instead of only accuracy?**
Accuracy favours large classes. Macro-F1 averages the F1 of each class, and UAR averages the recall of each class. UAR is the standard metric in emotion-recognition challenges.

**How do you know the differences are real?**
McNemar's exact test on the paired out-of-fold predictions, with Holm correction for multiple comparisons.

---

## Features (Aryan)

**What are MFCCs?**
Mel-frequency cepstral coefficients. Take the spectrum of a short frame, map it to the mel scale (which matches human pitch perception), take the log, then apply a discrete cosine transform. The first coefficients describe the overall shape of the spectrum, which reflects the shape of the vocal tract.

**What are Δ and ΔΔ?**
The first and second time-derivatives of the MFCCs, which capture how fast the timbre changes.

**How do you get one vector per clip from frames?**
Each descriptor is computed on 93 ms frames every 23 ms, then summarised by its mean and standard deviation over the clip. That gives 190 numbers per clip.

**How is pitch measured?**
With pYIN, a probabilistic version of the YIN pitch tracker. For 13 clips pYIN marked every frame unvoiced, so for those we used YIN on the louder half of the frames and removed octave jumps.

**Why trim silence and peak-normalise?**
About half of each recording is silence, which would distort the averages. Peak normalisation removes microphone-gain differences. As a side effect, our energy features measure energy dynamics, not absolute loudness. We state this as a limitation.

---

## Solution 1: Mutual Information + SVM (Godwin)

**What is mutual information?**
How much knowing a feature reduces uncertainty about the label. It's zero if the two are independent. Unlike correlation, it also captures non-linear relationships. scikit-learn estimates it with a k-nearest-neighbour method (Kraskov 2004; Ross 2014).

**Why is it called a filter method?**
It scores each feature on its own, before and independently of the classifier. That makes it fast, but it ignores redundancy: two near-duplicate features both score high.

**How is k chosen?**
By inner cross-validation over k ∈ {10, 20, …, 190}, together with C and γ.

**What does the SVM do?**
It finds the boundary with the widest margin between classes. The RBF kernel lets that boundary curve. C controls how many training errors are tolerated, and γ controls how local the kernel is. Eight classes are handled with one-vs-one voting.

---

## Solution 2: RFE + Random Forest (Sarthak)

**How does RFE work?**
Train a Random Forest, rank the features by importance, remove the 10 weakest, retrain, and repeat. The order of removal is the ranking.

**Why is it a wrapper method?**
It uses a model's own judgement (the forest's importances) to decide which features stay.

**What is feature importance in a Random Forest?**
Mean decrease in impurity: how much the splits on a feature reduce Gini impurity, averaged over all trees. It is known to be biased when features are correlated, so RFE only partly removes redundancy.

**How does a Random Forest work?**
Many decision trees, each trained on a bootstrap sample of the clips and considering a random subset of features at each split. The forest predicts by majority vote, and averaging reduces variance.

**Which citation for RFE?**
Guyon et al. (2002) introduced it with SVMs; Granitto et al. (2006) used it with Random Forest importances, as we do.

---

## Solution 3: Genetic Algorithm + MLP (Praveet)

**How does the GA represent a solution?**
A 190-bit mask: 1 means the feature is used, 0 means it is dropped.

**What is the fitness?**
The mean macro-F1 of the MLP over the actor-wise inner folds, minus 0.02 × the fraction of features used. The penalty prefers smaller subsets when accuracy is similar.

**Explain selection, crossover, mutation and elitism.**
- **Selection:** tournament. Pick 3 random masks; the best becomes a parent.
- **Crossover:** uniform. Each bit of the child comes from either parent with probability ½.
- **Mutation:** each bit flips with probability 1.5/190, which adds diversity.
- **Elitism:** the 2 best masks are copied unchanged, so the best solution is never lost.

**When does it stop?**
After 20 generations, or earlier if the best fitness hasn't improved for 6 generations.

**Why use a GA instead of ranking?**
Ranking methods judge features one at a time or greedily. A GA searches whole subsets, so it can find combinations that work well together. The cost is many more model evaluations.

**What is an MLP?**
A feed-forward neural network: input → hidden layer(s) with ReLU → softmax output. It is trained with the Adam optimiser on cross-entropy loss, with L2 regularisation (α) and early stopping.

**Did you write the GA yourselves?**
Yes, in `src/ser/ga_selector.py` in about 130 lines of NumPy, so every step can be explained.

---

## Likely tricky questions

- **"Your accuracy is lower than papers online that report 90%."** Those usually use random splits, where the same actors appear in training and testing. Our leakage experiment (notebook 06, §7) reproduces that inflation on our own data.
- **"Which solution is best, and is the difference significant?"** See the McNemar table in notebook 06, §3. Quote the Holm-adjusted p-values.
- **"Each solution changes both the selector and the classifier. Isn't that unfair?"** Each solution is compared with the same classifier on all features, and the 2 × 2 swap (MI + RF, RFE + SVM) in notebook 06, §6 separates the two effects.
- **"Is the selection stable?"** See the mean pairwise Jaccard index per method (notebook 06, §5).
- **"Limitations?"** Acted speech (not spontaneous), one language, peak normalisation removes absolute loudness, and only hand-crafted features.
