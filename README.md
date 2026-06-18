# ML-Assisted Tuning of Nonlinear PID (N-PID) Controllers

A high-performance controls and machine learning project implementing, optimizing, and predicting gains for **Nonlinear PID (N-PID)** controllers compared to traditional **Linear PID (L-PID)** baselines. 

This work simulates an armature-controlled DC motor plant with non-smooth Coulomb friction, applies **Particle Swarm Optimization (PSO)** to compute tracking-optimal controller parameters, and trains **multi-output machine learning models** to predict controller gains near-instantly directly from physical system settings.

**Theoretical Basis:** Çelik et al., "Advancements in nonlinear PID controllers: A comprehensive review," *Computers and Electrical Engineering*, 2026.

---

## Key Features

* **Continuous Plant Simulation:** Armature-controlled DC motor speed loop modeling Coulomb friction via a differentiable, tanh-smoothed approximation ($T_c \cdot \tanh(\omega/\epsilon)$).
* **RK4 Numerical Solver:** High-speed, inlined 4th-order Runge-Kutta solver delivering a **63x execution speedup** over general-purpose Python integration wrappers, enabling fast offline optimization sweeps.
* **Nonlinear Error-Shaping:** N-PID implementation modifying gains dynamically via proportional $\alpha(e)$, integral $\beta(e)$, and derivative $\gamma(de/dt)$ scaling.
* **Automated Ziegler-Nichols (ZN) Tuning:** Auto-detects critical gain $K_{cr}$ and oscillation period $T_{cr}$ via closed-loop binary search sweeps.
* **PSO Gain Optimization:** ITAE-based swarm tuning finding optimal damping characteristics.
* **Multi-Output Regression:** Random Forest and MLP networks predicting controller parameters in parallel.
* **Noise Sensitivity & Extrapolation Checks:** Robustness test suite comparing controllers under measurement feedback noise and out-of-distribution physical plant extrapolation parameters.

---

## Project Structure

```
npid-ml-tuning/
├── src/
│   ├── plant.py            # DC motor dynamics & open-loop verification (Section 1)
│   ├── controllers.py      # L-PID and N-PID controllers with anti-windup (Section 2)
│   ├── simulate.py         # RK4 closed-loop control simulation (Section 1.5)
│   ├── metrics.py          # Standard performance metrics calculations (Section 3.2)
│   ├── tuning_zn.py        # Ziegler-Nichols critical-gain auto-tuning (Section 3.1)
│   ├── tuning_pso.py       # Swarm optimizer for ITAE minimization (Section 3.3)
│   ├── generate_dataset.py # Batch dataset generator using parallel CPU pools (Section 3.4)
│   └── ml_models.py        # RF and MLP regressors training & speed testing (Section 4)
├── run_week1.py            # Week 1 baseline simulation and ZN tuning runner
├── run_week4.py            # Week 4 noise robustness and extrapolation runner
├── data/
│   └── tuning_dataset.csv  # 500-sample generated PID tuning database
├── results/
│   ├── figures/            # Saved system plots (step response, noise, etc.)
│   └── tables/             # CSV master benchmarking tables
├── report/
│   └── final_report.md     # Detailed 4-week project internship report
├── requirements.txt        # Project dependencies
└── README.md               # Quick start & results guide
```

---

## Installation & Setup

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd npid-ml-tuning
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Run Pipeline

### Week 1: Plant Verification & Ziegler-Nichols Baselines
Run the open-loop test, compute Ziegler-Nichols ultimate gains, and perform baseline comparisons:
```bash
python run_week1.py
```
*Outputs: `results/figures/open_loop_response.png`, `step_response_comparison.png`, `disturbance_rejection_comparison.png`, and `results/tables/week1_metrics.csv`.*

### Week 2: Particle Swarm Optimization (PSO)
Verify that the PSO tuning minimizes ITAE and outperforms Ziegler-Nichols:
```bash
python src/tuning_pso.py
```
*Outputs: Prints comparison metrics and saves `results/figures/pso_vs_zn_comparison.png`.*

### Week 2 (Cont.): Parallel Dataset Generation
Generate the 500-sample training dataset of physical configurations to optimal gains:
```bash
python src/generate_dataset.py --samples 500 --workers 4
```
*Outputs: Saves `data/tuning_dataset.csv`.*

### Week 3: ML Model Training & Benchmarks
Train Random Forest and MLP Multi-Output Regressors, run downstream stability checks, and run execution timing benchmarks:
```bash
python src/ml_models.py
```
*Outputs: Prints training metrics, speeds, and saves `results/figures/feature_importances.png`.*

### Week 4: Noise Robustness, Extrapolation & Master Comparison
Simulate the controllers under Gaussian feedback noise, test out-of-distribution extrapolation ($J = 0.03$), and generate master benchmarking tables:
```bash
python run_week4.py
```
*Outputs: Saves master tables in `results/tables/` and comparative plots in `results/figures/`.*

---

## 📊 Summary of Results

### 1. Baseline Performance (No Noise)
Evaluated on the default plant ($J=0.01, B=0.1, T_c=0.05$):

| Controller | Rise Time (s) | Overshoot (%) | Settling Time (s) | IAE | ITAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **L-PID (ZN)** | 0.1350 | 0.22% | 0.1990 | 2.3403 | 0.1631 |
| **N-PID (ZN)** | 0.1340 | 8.07% | 0.7510 | 3.5882 | 2.2720 |
| **N-PID (PSO)** | 0.1350 | 7.17% | 0.6480 | 2.7699 | 0.3466 |
| **N-PID (ML)** | 0.1350 | 7.20% | 0.6460 | 2.8060 | 0.3894 |

### 2. Feedback Noise Robustness ($\sigma = 0.3$)
Injected noise (~1.5% of reference speed $\omega_{ref} = 20\text{ rad/s}$) to test noise immunity:

| Controller | Rise Time (s) | Overshoot (%) | Settling Time (s) | IAE | ITAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **L-PID (ZN)** | 0.2620 | **54.18%** | 3.9990 | 61.5929 | **121.7759** |
| **N-PID (ZN)** | 0.1330 | 9.47% | 3.9220 | 3.6429 | 2.4825 |
| **N-PID (PSO)** | 0.1330 | 9.42% | 3.9200 | 3.6524 | 2.4984 |
| **N-PID (ML)** | 0.1330 | 9.69% | 3.9260 | 3.6880 | 2.5603 |

> 💡 **Key Takeaway:** Traditional L-PID performance collapses under feedback noise because high-frequency noise spikes propagate through the linear derivative path, injecting extreme voltages. **N-PID variants show remarkable noise immunity**, maintaining overshoot under **9.7%** and ITAE under **2.60**. This is due to the nonlinear derivative factor $\gamma(de/dt)$ which suppresses derivative gain automatically in response to large error rate-of-change spikes.

### 3. Out-of-Distribution Extrapolation ($J = 0.03$, Training Limit $J=0.02$)
Tested extrapolation limit under step load disturbance applied at $t=1.5\text{s}$:

| Controller | Rise Time (s) | Overshoot (%) | Settling Time (s) | IAE | ITAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **N-PID (PSO)** | 0.2090 | 10.83% | 1.2490 | 5.0541 | 1.3410 |
| **N-PID (ML)** | 0.2090 | 10.84% | 1.3670 | 5.0941 | 1.4124 |

* The ML regressor generalizes extremely well to out-of-distribution parameters, predicting parameters that match the true offline PSO search performance within a tiny fraction.

### 4. ML Speedup
* **PSO Optimization search time:** $30.108\text{ seconds}$
* **ML Regressor prediction time:** $15.95\text{ milliseconds}$
* **Performance gain:** **ML predicts optimal gains 1887.1x faster** than online iterative optimization, making it suitable for real-time control adjustments.

---

## 🔮 Future Work: Reinforcement Learning
A natural next step to extend this project's offline ML approach is **Online Adaptive Reinforcement Learning** (e.g., DDPG or TD3). An RL agent can learn directly from physical error feedback and rewards in real-time, adapting PID gains dynamically to handle parameter drift (e.g., motor heating increasing resistance $R$) without requiring prior parameter estimates.
