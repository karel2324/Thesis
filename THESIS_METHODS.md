# Optimizing Renal Replacement Therapy Initiation in Critically Ill Patients: A Reinforcement Learning Approach

## Methods

### 2.1 Study Design and Data Sources

This study employed offline reinforcement learning (RL) to develop and validate a decision support system for optimal timing of renal replacement therapy (RRT) initiation in critically ill patients with acute kidney injury (AKI). We utilized two large, publicly available intensive care unit (ICU) databases:

1. **Amsterdam University Medical Centers database (AUMCdb)**: A single-center Dutch ICU database containing admissions from 2003-2016, structured according to the Observational Medical Outcomes Partnership (OMOP) Common Data Model.

2. **Medical Information Mart for Intensive Care IV (MIMIC-IV)**: A multi-center US database from Beth Israel Deaconess Medical Center containing ICU admissions from 2008-2019.

The dual-database approach enabled internal validation (train/test split within each database) and external validation (training on one database, testing on the other).

### 2.2 Cohort Selection

Patients were included if they met the following criteria:
- Age ≥ 18 years at ICU admission
- KDIGO stage ≥ 2 (creatinine criterion) within the observation window
- ICU stay ≥ 24 hours

Patients were excluded if:
- RRT was initiated prior to reaching KDIGO stage 2
- Chronic kidney disease stage 5 (pre-existing dialysis)
- ICU stay < 24 hours
- Missing key outcome data

The index time (t₀) was defined as the timestamp when the patient first reached KDIGO stage 2 criteria. This served as the starting point for the decision-making process.

### 2.3 Markov Decision Process Formulation

We formulated the RRT initiation problem as a Markov Decision Process (MDP) with the following components:

#### 2.3.1 State Space
The state space consisted of 40-60 features (depending on MDP configuration) extracted at each time step, categorized as:

**Clinical Variables:**
- Vital signs: heart rate, mean arterial pressure, temperature, respiratory rate, SpO₂
- Laboratory values: creatinine, urea, potassium, sodium, lactate, pH, bicarbonate, hemoglobin, platelets, bilirubin
- Renal-specific: urine output (24h), creatinine trajectory (48h relative increase), KDIGO stage
- Organ function scores: SOFA components (cardiovascular, respiratory, coagulation, hepatic, neurological, renal)

**Derived Features:**
- Time since measurements (hours since last observation per variable)
- Missing value indicators (binary flags)
- Temporal features (hours since t₀)

All continuous variables were standardized using z-score normalization (μ=0, σ=1) fitted on the training set and applied to validation/test sets.

#### 2.3.2 Action Space
Binary action space: A = {0, 1}
- Action 0: Do not initiate RRT
- Action 1: Initiate RRT

#### 2.3.3 Time Discretization
Time was discretized into 8-hour intervals, creating a grid from t₀ to the terminal state. At each grid point, the action represents the decision of whether to initiate RRT in the subsequent 8-hour window.

#### 2.3.4 Reward Function
The reward function was designed to capture both short-term physiological improvement and long-term survival outcomes:

**Terminal Reward (ICU-Free Days):**
$$R_{terminal} = \frac{(H - ICU_{days}) \cdot \gamma^{steps}}{H}$$

Where:
- H = observation horizon (28 days)
- ICU_days = number of days in ICU within horizon
- γ = discount factor
- steps = number of time steps from terminal state to ICU discharge

For patients who died within the observation horizon:
- ICU-free days set to 0
- Optional mortality penalty: -1 (normalized)

**Intermediate Reward (SOFA Change):**
$$R_{intermediate} = \frac{\Delta SOFA}{SOFA_{max}} \cdot scale$$

Where:
- ΔSOFA = SOFA(t) - SOFA(t+24h)
- SOFA_max = 24 (maximum possible SOFA score)
- scale = configurable scaling factor

**Combined Reward:**
$$R_{total} = R_{intermediate} + R_{terminal}$$

### 2.4 Data Preprocessing Pipeline

#### 2.4.1 Missing Data Handling
Missing data was handled through a multi-stage imputation strategy:

1. **Clinical Imputation**: Variables with clinically meaningful default values (e.g., vasopressor dose = 0 when not administered, mechanical ventilation = 0 when not on ventilator)

2. **Time Feature Imputation**: Variables representing "time since last measurement" were imputed with a large value (168 hours) indicating no recent measurement.

3. **Multiple Imputation by Chained Equations (MICE)**: Remaining missing values were imputed using iterative imputation with Bayesian Ridge regression. Predictor variables included:
   - Clinical variables without missing values
   - Hours since t₀ (temporal context)
   - Gender (encoded as binary)

#### 2.4.2 Feature Scaling
Continuous variables were standardized using StandardScaler:
$$X_{scaled} = \frac{X - \mu}{\sigma}$$

Binary variables (including missing indicators) were not scaled to preserve interpretability.

### 2.5 MDP Configurations (Ablation Study)

To systematically evaluate the impact of feature sets and reward structures, we created five MDP configurations:

| MDP | Features | Reward | Purpose |
|-----|----------|--------|---------|
| MDP1 | Base | Terminal | Minimal state, sparse reward |
| MDP2 | Base | Combined | Minimal state, dense reward |
| MDP3 | Base + Missing Indicators | Terminal | Informative missingness |
| MDP4 | Base + Time Features | Terminal | Temporal context |
| MDP5 | All | Combined | Full model |

### 2.6 Reinforcement Learning Algorithms

#### 2.6.1 Conservative Q-Learning (CQL)
CQL was selected as the primary algorithm due to its conservative value estimation, which is crucial for healthcare applications where overestimation can lead to harmful recommendations. CQL adds a regularization term to the standard Q-learning objective:

$$\mathcal{L}_{CQL} = \alpha \cdot \mathbb{E}_{s \sim D}\left[\log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim \pi_\beta}[Q(s,a)]\right] + \mathcal{L}_{TD}$$

Where α controls the conservatism penalty.

#### 2.6.2 Comparison Algorithms
- **Double DQN (DDQN)**: Addresses overestimation bias through double Q-learning
- **Batch-Constrained Q-learning (BCQ)**: Constrains actions to those likely under the behavior policy
- **Neural Fitted Q-Iteration (NFQ)**: Batch Q-learning with neural networks

#### 2.6.3 Behavior Cloning Baseline
Behavior Cloning (BC) was trained to imitate the clinician's policy directly:
$$\mathcal{L}_{BC} = -\mathbb{E}_{(s,a) \sim D}[\log \pi_{BC}(a|s)]$$

This serves as a reference for the observed clinical practice.

### 2.7 Hyperparameter Optimization

Grid search was performed over the following hyperparameter space:

**CQL:**
- Learning rate: {1e-4, 3e-4, 1e-3}
- α (conservatism): {0.1, 1.0, 4.0, 10.0}
- Number of critics: {1, 2}
- Batch size: {256, 512}

**Network Architecture:**
- Hidden layers: [256, 256] (fully connected)
- Activation: ReLU
- Discount factor γ: 0.99

Training was performed for 30,000 gradient steps with evaluation every 2,000 steps.

### 2.8 Policy Evaluation

#### 2.8.1 Fitted Q-Evaluation (FQE)
Since online evaluation is not possible in the offline setting, we employed Fitted Q-Evaluation to estimate the expected value of learned policies:

$$V^{\pi}(s_0) = \mathbb{E}_{a \sim \pi}[Q^{\pi}(s_0, a)]$$

FQE trains a separate Q-function that evaluates the target policy using transitions from the behavior policy.

#### 2.8.2 Bootstrap Confidence Intervals
Following Hao et al. (2022), we computed 95% confidence intervals through bootstrapping:

1. Train FQE on original dataset → V_original
2. For b = 1 to B:
   - Sample episodes with replacement
   - Train FQE on bootstrap sample → V_b
   - Compute error: ε_b = V_b - V_original
3. CI = [V_original - q_97.5(ε), V_original - q_2.5(ε)]

#### 2.8.3 Evaluation Metrics
- **Initial State Value (ISV)**: FQE-estimated expected return from initial states
- **Action Match**: Proportion of states where π_learned(s) = π_clinician(s)
- **TD Error**: Temporal difference error on validation set
- **RRT Rate**: Proportion of states where action = 1

### 2.9 External Validation

External validation assessed generalizability by:
1. Training models on AUMCdb (Dutch single-center)
2. Evaluating on MIMIC-IV (US multi-center)

Key considerations:
- Same scaler applied (fitted on training database)
- Same MDP structure
- Comparison with locally-trained Behavior Cloning

### 2.10 Statistical Analysis

- Continuous variables: median [IQR] or mean ± SD
- Categorical variables: n (%)
- Between-group comparisons: Wilcoxon rank-sum test, chi-square test
- Policy value comparisons: Bootstrap 95% CIs with 100 resamples
- Significance level: α = 0.05

### 2.11 Implementation

All analyses were performed using:
- Python 3.11
- d3rlpy 2.x (offline RL algorithms)
- scikit-learn (imputation, scaling)
- pandas, numpy (data manipulation)
- PyTorch (neural networks, CUDA acceleration)
- BigQuery SQL (data extraction)

Code is available at: [repository URL]

---

## Expected Results Structure

### 3.1 Cohort Characteristics
- Table 1: Baseline characteristics (AUMCdb vs MIMIC)
- Figure 1: Patient flow diagram

### 3.2 Ablation Study
- Table 2: MDP configuration comparison (ISV, Action Match)
- Figure 2: Training curves per MDP

### 3.3 Algorithm Comparison
- Table 3: HPO results per algorithm
- Figure 3: FQE comparison (CQL vs BC vs Clinician)

### 3.4 Policy Analysis
- Figure 4: When does learned policy recommend RRT vs clinician?
- Table 4: Patient subgroups with highest policy divergence

### 3.5 External Validation
- Table 5: Cross-database performance
- Figure 5: Generalizability assessment

---

## Discussion Points

### Strengths
1. **Dual-database validation**: Both internal and external validation
2. **Conservative algorithm**: CQL prevents value overestimation
3. **Clinically meaningful reward**: ICU-free days + mortality
4. **Transparent methodology**: Config-driven, reproducible pipeline
5. **Multiple MDP configurations**: Systematic feature importance assessment

### Limitations
1. **Offline setting**: Cannot verify counterfactual outcomes
2. **Action simplification**: Binary action (RRT timing, not modality/dose)
3. **Database differences**: Practice patterns, patient populations
4. **Unmeasured confounding**: Cannot fully account for clinical judgment

### Clinical Implications
[To be completed based on results]

---

## References

1. Hao et al. (2022). Bootstrapping FQE for Offline Policy Evaluation
2. Kumar et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning
3. Komorowski et al. (2018). AI Clinician for Sepsis Treatment
4. STARRT-AKI Investigators (2020). Timing of RRT in AKI
5. Gaudry et al. (2016). AKIKI Trial

