# Classifying Songs By Plot Archetypes Using Logistic Regression

I use binary classification and multinomial softmax models to classify songs by artist (Olivia Rodrigo or Gracie Abrams) and album. For the binary classifier (the basic logistic regression model), features with strong positive coefficients define Gracie Abrams' style, and features with strong negative coefficients define Olivia Rodrigo's style. In the multinomial softmax model, each of the six albums is modeled with its own linear score equation ($z_k = \mathbf{w}_k^T \mathbf{x} + b_k$), and positive coefficients indicate that a feature makes a song more likely to belong to that specific album (and vice versa). The features in this data are how well each song fits each of the seven basic plot archetypes. 

[Quick comment on results]

For future work, I will try to use a K-Means classifier on a larger dataset of songs from more artists to see which artists like to write which kinds of songs. Each cluster will be interpretable based on the seven basic plot archetypes, and I will analyze how many songs from each artist are in each cluster. 

---

## Experimental Design

This experiment consists of two phases. Phase one is a binary classification of the songs by artist (0 = Olivia Rodrigo, 1 = Gracie Abrams). Here is the equation:
```math
P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
```
\
The feature vector w has seven dimensions, each corresponding to a rating of how well the song fits one of the seven plot archetypes (explained in further detail below). These features are transformed to follow a standard normal distribution so we can compare the coefficients stored in w. The goal is to find the specific features that are the most helpful in classifying each artist. The magnitude of each coefficient is a proxy for explanatory power, and a negative sign indicates that feature best describes lyrics by Olivia Rodrigo (and positive indicates Gracie Abrams).  

Phase two complicates this model by breaking down the classes into the six albums reviewed in the dataset. Thus, we will get six different sets of coefficients: 
```math
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{6} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
```
\
In this case, positive coefficients mean the presence of that feature makes it more likely that a song belongs to that class. This makes the raw numbers harder to interpret. I will have to compare the values of the features for all six sets of coefficients, looking for the highest positive coefficients to mean the song has the best chance of being in that album. Still, that chance could very well be less than 50%, and some albums might be very similar in terms of the archetypes involved, so this might be tricky to interpret. 

---

## Explanation of Raw Data

The dataset, artist_comparison.xlsx, contains 90 songs (41 by Olivia Rodrigo, and 49) by Gracie Abrams. The albums included are SOUR, GUTS, You Seem Pretty Sad For A Girl So In Love, Good Riddance, The Secret Of Us, and Daughter From Hell. Each observation has the song name, an artist label, album label, and scores of fit to each of the seven basic plot archetypes. An explanation of the scoring system I used is included in "rating_specification" sheet, and also here:

### The Seven Basic Plot Structures
1) Overcoming the Monster
This applies to songs where the speaker battles an external villain, most commonly their ex or some facet of social norms / society. A perfect fit (score 3/3) has the speaker overcome the villain in addition to venting about them. 

2) Rags to Riches
In the context of a song, rags to riches refers to a story of the speaker experiencing a period of good fortune that helps them overcome some form of external adversity (any proxy for poverty, for instance). A perfect fit includes a marked change in the speaker's worldview after experiencing 'riches'.

3) The Quest
A quest refers to a journey that is deliberately undertaken with the goal of finding something specific, whether that's a material treasure or a feeling. A perfect fit focuses on how the journey changes the speaker.

4) Voyage and Return
By contrast, a voyage and return is more of an accidental or life experience journey that had no specific aim. A perfect fit focuses on how undertaking and returning from the journey inspired the speaker to make some change in their life. 

5) Comedy
This is one of the more self-explanatory and clear-cut archetypes. A comedic song is a song that invokes laughter and uses humor/satire to deliver a message. A perfect fit leans into mocking some trope of society. 

6) Tragedy
A tragic song is about succumbing to an internal villain, and has the speaker lament their inability to overcome it. This is what most sad songs follow, but a perfect fit emphasizes that the speaker regrets who they've become in some way. 

7) Rebirth
In some ways, this is the sequel to or the opposite of a tragedy. The speaker overcomes internal adversity to reinvent themselves. A perfect fit gives the speaker some form of redemption after the positive transformation (like a comeback story).

### Prediction
I stored the data in a google sheet, and used Gemini to analyze trends to gain some level of intuition before running the regression. 
\
![Predicted trends](gabrams_orodrigo_trends.png)

###
---

## Results & Analysis

Here were my results from running the full experiment:

### 1. Artist Classification

Here is the learned logistic regression equation:
```math
P(y=1 \mid \mathbf{x}) = \sigma(- 0.0246 \cdot x_{\text{Monster}} - 0.3605 \cdot x_{\text{RagsRiches}} + 0.1741 \cdot x_{\text{Quest}} - 0.2358 \cdot x_{\text{VoyageReturn}} - 1.1735 \cdot x_{\text{Comedy}} + 0.7035 \cdot x_{\text{Tragedy}} + 0.6876 \cdot x_{\text{Rebirth}} + 0.0614)
```
\
In this case, Comedy was a very strong indicator that a song was written by Olivia Rodrigo, which matches our expectation. Similarly, the Tragedy and Rebirth archetypes hinted that a song was written by Gracie Abrams. Surprisingly, however, the Overcoming the Monster plot was the least helpful predictor, despite being more common in Olivia Rodrigo's songs. 
 
---

### 2. Album Classification

Here are the six learned logistic regression equations:
```math
\begin{aligned}
z_{\text{SOUR}} = 0.2906 \cdot x_{\text{Monster}} + 0.6389 \cdot x_{\text{RagsRiches}} - 0.5459 \cdot x_{\text{Quest}} - 0.2266 \cdot x_{\text{VoyageReturn}} - 0.0861 \cdot x_{\text{Comedy}} - 0.8630 \cdot x_{\text{Tragedy}} - 0.9558 \cdot x_{\text{Rebirth}} - 0.6280 \\
z_{\text{GUTS}} = 0.0890 \cdot x_{\text{Monster}} - 0.0050 \cdot x_{\text{RagsRiches}} + 0.2949 \cdot x_{\text{Quest}} + 0.3786 \cdot x_{\text{VoyageReturn}} + 0.9604 \cdot x_{\text{Comedy}} + 0.1509 \cdot x_{\text{Tragedy}} - 0.0412 \cdot x_{\text{Rebirth}} + 0.3912 \\
z_{\text{You seem pretty sad for a girl so in love}} = - 0.2922 \cdot x_{\text{Monster}} + 0.0133 \cdot x_{\text{RagsRiches}} + 0.0680 \cdot x_{\text{Quest}} + 0.1162 \cdot x_{\text{VoyageReturn}} + 0.6658 \cdot x_{\text{Comedy}} - 0.4081 \cdot x_{\text{Tragedy}} - 0.0998 \cdot x_{\text{Rebirth}} + 0.1296 \\
z_{\text{Good Riddance}} = - 0.2418 \cdot x_{\text{Monster}} - 0.2514 \cdot x_{\text{RagsRiches}} + 0.6727 \cdot x_{\text{Quest}} + 0.0614 \cdot x_{\text{VoyageReturn}} - 0.5826 \cdot x_{\text{Comedy}} + 0.5479 \cdot x_{\text{Tragedy}} + 0.3294 \cdot x_{\text{Rebirth}} - 0.0308 \\
z_{\text{The Secret Of Us}} = - 0.1592 \cdot x_{\text{Monster}} - 0.7316 \cdot x_{\text{RagsRiches}} + 0.1071 \cdot x_{\text{Quest}} + 0.2629 \cdot x_{\text{VoyageReturn}} - 0.2019 \cdot x_{\text{Comedy}} - 0.0645 \cdot x_{\text{Tragedy}} + 0.4632 \cdot x_{\text{Rebirth}} + 0.2067 \\
z_{\text{Daughter From Hell}} = 0.3137 \cdot x_{\text{Monster}} + 0.3358 \cdot x_{\text{RagsRiches}} - 0.5967 \cdot x_{\text{Quest}} - 0.5925 \cdot x_{\text{VoyageReturn}} - 0.7556 \cdot x_{\text{Comedy}} + 0.6369 \cdot x_{\text{Tragedy}} + 0.3041 \cdot x_{\text{Rebirth}} - 0.0687
\end{aligned}
```
\
In this case, we can make a few observations about the archetype profile of each album. SOUR is mainly characterized by the absence of tragedy, since its songs lean heavier into an external v.s. internal villain. GUTS leans more into comedy, a distinguishing feature from the other albums. Olivia's third album has this to a lesser extent, and still followed her trend of external rather than internal villains. On the other hand, Good Riddance focuses on Quest, Tragedy, and Rebirth, which makes sense for Gracie's introspective writing style. The Secret Of Us is characterized by lack of Rags to Riches plots, meaning Gracie doesn't tend to rewrite about good fortune, and her Rebirth insights tend to involve more regret than gained confidence. However, in her most recent album, Daughter From Hell, Gracie finally leans into the Monster plot, and away from Quest. Her lyrics are about more unintentional experience with her exes. However, she does so without the use of Comedy, a key difference from Olivia. This mostly matches the predictions from visually inspecting the data. 
 
---

### 3. Robustness of Artist Classification

To validate that the results from part (1) were accurate, I ran phase 1 over 25 replications and reported 90% confidence intervals for each of the regression coefficients. Before each trial, the order of observations in the dataset was shuffled. 
```math
\begin{aligned}
w_{\text{Monster}} &= -0.0246 \quad (90\% \text{ CI}: [-0.4862, 0.4370]) \\
w_{\text{RagsRiches}} &= -0.3605 \quad (90\% \text{ CI}: [-0.8241, 0.1030]) \\
w_{\text{Quest}} &= 0.1741 \quad (90\% \text{ CI}: [-0.3041, 0.6523]) \\
w_{\text{VoyageReturn}} &= -0.2358 \quad (90\% \text{ CI}: [-0.7022, 0.2306]) \\
w_{\text{Comedy}} &= -1.1735 \quad (90\% \text{ CI}: [-1.9056, -0.4413]) \\
w_{\text{Tragedy}} &= 0.7035 \quad (90\% \text{ CI}: [0.2152, 1.1918]) \\
w_{\text{Rebirth}} &= 0.6876 \quad (90\% \text{ CI}: [0.2109, 1.1642]) \\
b_{\text{Intercept}} &= 0.0614 \quad (90\% \text{ CI}: [-0.3863, 0.5092])
\end{aligned}
```
\
Comedy is confirmed as the most reliable indicator of Olivia Rodrigo's songs, with the entire interval being strongly negative. Inversely, the intervals for Tragedy and Rebirth are positive, telling us that Gracie Abrams leans more into those archetypes. Unsurprisingly, many of the other archetypes had intervals that included 0, which tell us that both artists use them and they don't have much predictive power. However, the one surprising result here is that Overcoming the Monster was not a useful predictor for Olivia Rodrigo's songs. In our precursory analysis, we noted that Olivia tends to lean more into external villains while Gracie laments inward. Yet, Gracie's latest album did shift more into the Monster plot, and Olivia's newer albums experimented with comedy, among other new plots, which likely added some noise to the data. 

---

### 4. Robustness of Album Classification

To validate that the results from part (2) had some inherent meaning, I ran phase 2 over 25 replications. There are too many coefficients to report confidence intervals in a reader-friendly way, so instead I will make note of any significant sources of variation between the trials. Once again, the dataset was scrambled before each trial. 

Across the 25 scrambled replications, the empirical variance of the fitted coefficients was virtually zero ($\text{std} \approx 10^{-16}$). This is an expected mathematical property of the L-BFGS optimization algorithm: because L-BFGS evaluates the multinomial cross-entropy loss over the complete dataset, full-batch convex optimization is strictly invariant to sample ordering. Consequently, observation shuffling confirms algorithmic stability, but empirical trial-to-trial variance would require bootstrap resampling (sampling with replacement) or cross-validation subsampling rather than row permutations.

However, examining the **inter-album variability** of the learned coefficients across the six albums reveals the primary plot archetypes driving thematic differentiation:

| Plot Archetype | Min Coefficient | Max Coefficient | Coefficient Range | Inter-Album Std | Highest Associating Album | Lowest Associating Album |
|---|---|---|---|---|---|---|
| **Comedy** | -0.7556 | +0.9604 | **1.7160** | **0.6224** | *GUTS* (+0.9604) | *Daughter From Hell* (-0.7556) |
| **Tragedy** | -0.8630 | +0.6369 | **1.4999** | **0.5235** | *Daughter From Hell* (+0.6369) | *SOUR* (-0.8630) |
| **Rebirth** | -0.9558 | +0.4632 | **1.4189** | **0.4727** | *The Secret Of Us* (+0.4632) | *SOUR* (-0.9558) |
| **Rags to Riches** | -0.7316 | +0.6389 | **1.3704** | **0.4319** | *SOUR* (+0.6389) | *The Secret Of Us* (-0.7316) |
| **The Quest** | -0.5967 | +0.6727 | **1.2694** | **0.4490** | *Good Riddance* (+0.6727) | *Daughter From Hell* (-0.5967) |
| **Voyage and Return** | -0.5925 | +0.3786 | **0.9711** | **0.3246** | *GUTS* (+0.3786) | *Daughter From Hell* (-0.5925) |
| **Overcoming the Monster** | -0.2922 | +0.3137 | **0.6059** | **0.2450** | *Daughter From Hell* (+0.3137) | *You seem pretty sad...* (-0.2922) |
| *Intercept (Base Rate)* | -0.6280 | +0.3912 | 1.0192 | 0.3195 | *GUTS* (+0.3912) | *SOUR* (-0.6280) |

#### Key Sources of Thematic Variability:
1. **Comedy is the Single Largest Source of Album Divergence (Range: 1.7160, Std: 0.6224)**:
   Comedy exhibits the widest spread of any feature in the model. It sharply splits the discographies: it strongly defines Olivia Rodrigo's sophomore release *GUTS* (+0.9604) and EP (+0.6658), but acts as an intense negative predictor for Gracie Abrams' serious, confessional projects *Daughter From Hell* (-0.7556) and *Good Riddance* (-0.5826).

2. **Tragedy and Rebirth Form the Emotional Separation Axis (Ranges: 1.4999 and 1.4189)**:
   Both Tragedy and Rebirth show wide divergence across the catalog. Gracie's *Daughter From Hell* (+0.6369) and *Good Riddance* (+0.5479) are heavily dominated by tragic introspection, whereas Olivia's debut *SOUR* strongly penalizes both Tragedy (-0.8630) and Rebirth (-0.9558), as its breakup anthems focus outward on external blame rather than internal self-destruction or renewal. Conversely, *The Secret Of Us* (+0.4632) is characterized most strongly by Rebirth.

3. **Rags to Riches vs. Quest Distinguish Specific Eras**:
   *SOUR* uniquely capitalizes on Rags to Riches (+0.6389), while *The Secret Of Us* actively rejects it (-0.7316). Meanwhile, *Good Riddance* stands out as a clear outlier on The Quest (+0.6727), reflecting deliberate exploratory emotional journeys that are notably absent from *Daughter From Hell* (-0.5967) and *SOUR* (-0.5459).

4. **Overcoming the Monster Shows the Lowest Variability (Range: 0.6059, Std: 0.2450)**:
   Across all six albums, the Monster archetype is the least differentiating feature, with its coefficients tightly constrained between -0.2922 and +0.3137. Battling an external adversary/ex is ubiquitous across pop breakup songwriting regardless of the specific album era, making it the weakest individual archetype for isolating a specific album.

5. **Base Rate Intercept Variability (Range: 1.0192)**:
   The baseline intercepts vary from -0.6280 (*SOUR*) to +0.3912 (*GUTS*). This spread primarily reflects album sample sizes in the dataset (*SOUR* has only 11 songs, while *GUTS* and *The Secret Of Us* have 17 songs each).

---

### 5. Conclusions & Discussion

[Comment on general accuracy and interpretability of results]

---

#### Changes from Original Specification 

[Report any changes here]

---

## Codebase Architecture

```
gabrams_orodrigo/
├── preprocess.py             # Load data, feature scaling
├── models.py                 # Define the regression models for both phases
├── stats.py                  # Tracks confidence intervals and variability across replications
├── experiment.py             # Orchestrates phase 1 and phase 2, including any outputs
├── runner.py                 # CLI interface for custom trials, rapid smoke tests, individual phases, and full runs
├── LogReg_SongClassifier.pdf # Original mathematical problem specification
├── artist_comparison.xlsx    # Original dataset 
├── tests/                    # Automated unit and integration test suite
│   ├── test_models.py        # Tests the logic of the logistic regression models
│   ├── test_math.py          # Ensure that all of the calculations are mathematically correct, includes confidence intervals
│   ├── test_experiment.py    # Ensure that the pipeline runs correctly and outputs are in the correct format
│   └── test_smoke.py         # Rapid end-to-end integration smoke test
└── README.md                 # Project specification, experiment documentation, architecture, and usage
```

---

## Installation & Usage

### Prerequisites
The codebase requires Python 3.10+ and the following packages:
```bash
pip install numpy pandas scikit-learn scipy openpyxl
```

### Run unittests
Run the complete automated test suite (16 unit and integration tests):
```bash
python3 -m unittest discover -s tests
```

### Run a quick smoke test
Verify that the experiment runs end-to-end with a rapid integration test:
```bash
python3 runner.py --smoke-test
```

### Run the experiment

#### 1. Run Full Experiment
```bash
python3 runner.py --phase all
```

#### 2. Run Individual Phases
- **Phase 1 Only** (Binary classification of songs by artist: Olivia Rodrigo vs Gracie Abrams):
  ```bash
  python3 runner.py --phase 1 --trials 25
  ```

- **Phase 2 Only** (Multinomial classification of songs across the 6 albums):
  ```bash
  python3 runner.py --phase 2 --trials 25 
  ```

### Command-Line Arguments Reference
| Argument | Description | Default |
|---|---|---|
| `--smoke-test` | Run a rapid end-to-end integration check across both phases | `False` |
| `--phase [1\|2\|all]` | Run a specific phase or all phases in sequence | `None` |
| `--trials N` | Number of replications of each regression | `1` |
| `--output PATH` | Path to export structured experimental results as JSON | `None` |
