# Project Report: ML-Assisted Tuning of Nonlinear PID (N-PID) Controllers

**Internship Project Report (Weeks 1 - 4)**  
**Author:** Final-year B.Tech ECE Student  
**Task:** Design, simulate, and benchmark Nonlinear PID (N-PID) controllers, implement Particle Swarm Optimization (PSO) auto-tuning, and train Machine Learning (ML) regressors for instant, robust controller gain prediction on a DC motor system.

---

## 1. Executive Summary

This project implements and evaluates the performance of a **Nonlinear PID (N-PID) controller** relative to a traditional **Linear PID (L-PID)** controller on an armature-controlled DC motor plant. The plant incorporates a non-smooth Coulomb friction nonlinearity, which creates stiction and tracking challenges for standard linear controllers. 

To overcome these, we:
1. Formulated the plant's continuous dynamics and solved them using inlined 4th-order Runge-Kutta (RK4) integration for rapid control simulation loops.
2. Built L-PID and N-PID controllers with standard back-calculation integral anti-windup.
3. Automated Ziegler-Nichols (ZN) ultimate gain tuning, establishing a baseline.
4. Implemented Particle Swarm Optimization (PSO) to tune the PID gains to minimize the Integral of Time-weighted Absolute Error (ITAE).
5. Generated a 500-sample dataset representing different physical motor inertias, frictions, disturbances, and reference speeds.
6. Trained a multi-output Random Forest Regressor to predict optimal gains ($K_p$, $K_i$, $K_d$) near-instantly, resulting in an **1887x speedup** over iterative online PSO search.
7. Validated the controllers under measurement noise and out-of-distribution extrapolation tests, demonstrating that the N-PID structures provide superior robustness—particularly due to the noise-filtering capabilities of the nonlinear derivative gain component $\gamma(de/dt)$.

---

## 2. Plant Physics & Modeling

The anchor system is an armature-controlled DC motor speed regulation loop. The system contains electromechanical coupling characterized by two coupled differential equations:

### 2.1 Governing Equations

1. **Electrical Dynamics:**
   $$\frac{di(t)}{dt} = \frac{V(t) - R \cdot i(t) - K_e \cdot \omega(t)}{L}$$

2. **Mechanical Dynamics:**
   $$\frac{d\omega(t)}{dt} = \frac{K_t \cdot i(t) - B \cdot \omega(t) - T_{\text{load}}(t) - T_{\text{friction}}(t)}{J}$$

Where:
* $V(t)$ is the armature voltage control input ($u(t) \in [-V_{\text{max}}, V_{\text{max}}]$).
* $i(t)$ is the armature current.
* $\omega(t)$ is the angular velocity output ($y(t)$).
* $T_{\text{load}}(t)$ is the external disturbance step torque.
* $T_{\text{friction}}(t)$ is the non-smooth Coulomb friction torque.

### 2.2 Coulomb Friction Nonlinearity

Coulomb friction introduces a discontinuity at zero velocity:
$$T_{\text{friction}}(t) = T_c \cdot \text{sign}(\omega(t))$$

To prevent severe numerical stiff-equation integration errors or high-frequency chattering in `scipy.integrate.solve_ivp` and manual integration solvers, we implement a smoothed differentiable approximation:
$$T_{\text{friction}}(t) = T_c \cdot \tanh\left(\frac{\omega(t)}{\epsilon}\right)$$
where $\epsilon = 0.01$ is a small smoothing parameter.

### 2.3 Plant Parameters

The default physical constants used to verify the baseline step response are:

| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| Armature Resistance | $R$ | 1.0 | $\Omega$ |
| Armature Inductance | $L$ | 0.5 | H |
| Back-EMF Constant | $K_e$ | 0.5 | $\text{V}\cdot\text{s}/\text{rad}$ |
| Torque Constant | $K_t$ | 0.5 | $\text{N}\cdot\text{m}/\text{A}$ |
| Rotor Inertia | $J$ | 0.01 | $\text{kg}\cdot\text{m}^2$ |
| Viscous Friction Coefficient | $B$ | 0.1 | $\text{N}\cdot\text{m}\cdot\text{s}/\text{rad}$ |
| Coulomb Friction Torque | $T_c$ | 0.05 | $\text{N}\cdot\text{m}$ |
| Maximum Control Input Voltage | $V_{\text{max}}$ | 24.0 | V |

---

## 3. Controller Architectures

We implemented two primary control laws in `src/controllers.py`:

### 3.1 Linear PID (L-PID)

The baseline L-PID controller is represented by:
$$u(t) = K_p \cdot e(t) + K_i \int_{0}^{t} e(\tau)d\tau + K_d \cdot \frac{de(t)}{dt}$$
* **Integral Anti-Windup:** Back-calculation anti-windup clamps the accumulated error integral when the control voltage saturates at $\pm V_{\text{max}}$, preventing prolonged overshoot.

### 3.2 Nonlinear PID (N-PID)

The generalized N-PID modifies each control action with error-dependent gain multipliers (Eq. 9, Çelik et al., 2026):
$$u(t) = K_p \cdot \alpha(e(t)) \cdot e(t) + K_i \cdot \beta(e(t)) \int_{0}^{t} e(\tau)d\tau + K_d \cdot \gamma(e(t)) \cdot \frac{de(t)}{dt}$$

Where the nonlinear gain functions are defined as:

1. **Nonlinear Proportional Gain $\alpha(e)$:**
   $$\alpha(e) = \frac{e}{\left(1 + |e|^n\right)^{1/n}}, \quad n > 0 \quad (\text{Default } n=2)$$
   * *Physical impact:* Saturates proportional action for large errors to prevent overshoot, while remaining linear ($\alpha(e) \approx 1$) for small tracking errors.

2. **Nonlinear Integral Gain $\beta(e)$:**
   $$\beta(e) = \frac{1}{1 + |e|^m}, \quad m > 0 \quad (\text{Default } m=2)$$
   * *Physical impact:* Suppresses integral action when errors are large (transient phase) to prevent integrator windup, restoring standard integration ($\beta(e) \rightarrow 1$) as the system approaches steady state.

3. **Nonlinear Derivative Gain $\gamma(de/dt)$:**
   $$\gamma\left(\frac{de}{dt}\right) = \frac{1}{1 + \left|\frac{de}{dt}\right|^p}, \quad p > 0 \quad (\text{Default } p=2)$$
   * *Physical impact:* Intentionally dampens high-frequency rate-of-change spikes (such as measurement noise) by letting $\gamma \rightarrow 0$, preventing derivative kick and instability, while keeping normal damping properties intact near steady state.

---

## 4. Tuning Methodologies

### 4.1 Ziegler-Nichols (Baseline)

The Ziegler-Nichols tuning baseline is found by zeroing out $K_i$ and $K_d$ and raising $K_p$ until a sustained, constant-amplitude oscillation is observed.
* **Auto-Tuning Detection:** Sweep $K_p$ using a binary search to find the ultimate gain $K_{cr}$ where peak-to-peak amplitude remains constant over the final few oscillation cycles.
* **Convergence Result:** 
  * Ultimate Gain: $K_{cr} = 232.0000$
  * Ultimate Period: $T_{cr} = 0.0413\text{ s}$
  * Standard ZN gains: $K_p = 139.2000, K_i = 6747.3130, K_d = 0.7179$.

### 4.2 Particle Swarm Optimization (PSO)

To tune gains for optimal performance, a PSO optimizer from `pyswarms` was constructed.
* **Fitness Metric:** Integral of Time-weighted Absolute Error (ITAE) with an overshoot penalty:
  $$\text{Fitness} = \int_{0}^{t_{\text{end}}} t \cdot |e(t)| dt + w_{\text{overshoot}} \cdot \max(0, \% \text{ Overshoot} - 10)$$
* **Search Space:** $K_p \in [10, 500]$, $K_i \in [100, 15000]$, $K_d \in [0.05, 5.0]$.
* **Hyperparameters:** $N_{\text{particles}} = 20$, $N_{\text{iterations}} = 40$, $c_1=0.5$, $c_2=0.3$, $w=0.9$.
* **Results:** PSO finds optimal gains $K_p = 127.9988, K_i = 681.7335, K_d = 1.8942$, achieving a **85% reduction in ITAE** (from 2.272 to 0.341) compared to Ziegler-Nichols N-PID tuning.

---

## 5. Dataset Generation & Machine Learning Models

### 5.1 Dataset Generation
We generated a dataset of **500 trials** by uniform random sampling of motor parameters across physical boundaries:
* Inertia $J \in [0.005, 0.02] \text{ kg}\cdot\text{m}^2$
* Viscous friction $B \in [0.05, 0.2] \text{ N}\cdot\text{m}\cdot\text{s}/\text{rad}$
* Coulomb friction $T_c \in [0.02, 0.1] \text{ N}\cdot\text{m}$
* Load torque step disturbance $T_{\text{load}} \in [0.0, 0.05] \text{ N}\cdot\text{m}$ (applied at $t = 1.5\text{s}$)
* Reference speed $\omega_{\text{ref}} \in [5.0, 50.0] \text{ rad/s}$

For each configuration, parallelized PSO optimizations were executed to determine the optimal $K_p, K_i, K_d$ values.

### 5.2 Model Training
Using an 80/20 train/test split, we trained two multi-output regression models to map $[J, B, T_c, T_{\text{load}}, \omega_{\text{ref}}] \rightarrow [K_p, K_i, K_d]$:
1. **Random Forest Regressor (RF):** 200 estimators.
2. **Multi-Layer Perceptron (MLP):** Two hidden layers of shape $(32, 16)$, Relu activation, 2000 max iterations.

Input features were scaled via `StandardScaler`, and targets were left in physical units for direct output capability.

### 5.3 Training Results & Speed Benchmarks

* **Downstream Simulation Stability:** **100%** of the 100 test configurations remained stable when simulated using the Random Forest predicted gains.
* **Accuracy Metrics:**
  * Proportional Gain ($K_p$): Random Forest $R^2 \approx -0.01$, MLP $R^2 \approx -0.40$
  * Integral Gain ($K_i$): Random Forest $R^2 \approx 0.17$, MLP $R^2 \approx 0.15$
  * Derivative Gain ($K_d$): Random Forest $R^2 \approx 0.14$, MLP $R^2 \approx -0.49$
  * *Discussion:* Though the tabular regression $R^2$ is low due to the complex, highly non-convex nature of the PSO landscape (where multiple different combinations of gains yield similarly low ITAE costs), the downstream simulation performance is exceptional. The median ITAE ratio between the ML-predicted controller and the true PSO-optimal controller is **1.016**, proving the ML model predicts gains that act almost identically to the PSO-optimal settings.
* **Speedup:** 
  * PSO Search time: $30.108\text{ seconds}$
  * ML Inference time: $15.95\text{ milliseconds}$
  * **ML-predicted gains are 1887.1x FASTER** than iterative PSO searches.

---

## 6. Robustness and Generalization (Week 4 Results)

We benchmarked four controller variants:
1. **L-PID (ZN):** Standard linear PID, Ziegler-Nichols tuned.
2. **N-PID (ZN):** Nonlinear PID, Ziegler-Nichols tuned.
3. **N-PID (PSO):** Nonlinear PID, PSO tuned.
4. **N-PID (ML):** Nonlinear PID, Random Forest predicted gains.

### 6.1 Baseline Performance (No Noise)

| Controller | Rise Time (s) | Overshoot (%) | Settling Time (s) | IAE | ISE | ITAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L-PID (ZN)** | 0.1350 | 0.22% | 0.1990 | 2.3403 | 35.1934 | 0.1631 |
| **N-PID (ZN)** | 0.1340 | 8.07% | 0.7510 | 3.5882 | 36.1169 | 2.2720 |
| **N-PID (PSO)**| 0.1350 | 7.17% | 0.6480 | 2.7699 | 35.7913 | 0.3466 |
| **N-PID (ML)** | 0.1350 | 7.20% | 0.6460 | 2.8060 | 35.8032 | 0.3894 |

*Analysis:* In noise-free conditions, ZN-tuned Linear PID performs exceptionally well because the baseline linear equations are symmetrical. N-PID (PSO) and N-PID (ML) perform closely, with ML outputting gains that achieve an ITAE of 0.3894, nearly matching the PSO optimum of 0.3466.

### 6.2 Noise Robustness Test (Feedback Noise $\sigma = 0.3$)

Measurement noise $N(0, \sigma^2)$ representing 1.5% of the reference speed (20 rad/s) was injected into the feedback loop.

| Controller | Rise Time (s) | Overshoot (%) | Settling Time (s) | IAE | ISE | ITAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L-PID (ZN)** | 0.2620 | 54.18% | 3.9990 | 61.5929 | 953.5127 | 121.7759 |
| **N-PID (ZN)** | 0.1330 | 9.47% | 3.9220 | 3.6429 | 36.1434 | 2.4825 |
| **N-PID (PSO)**| 0.1330 | 9.42% | 3.9200 | 3.6524 | 36.1526 | 2.4984 |
| **N-PID (ML)** | 0.1330 | 9.69% | 3.9260 | 3.6880 | 36.2178 | 2.5603 |

*Analysis:* Under noise, the **L-PID performance collapses**, with overshoot exploding to **54.18%** and ITAE degrading to **121.77**. This occurs because high-frequency measurement noise propagates through the linear derivative path, injecting huge voltage fluctuations into the plant.  
In contrast, all **N-PID controllers display remarkable noise immunity**, maintaining overshoot under **9.7%** and keeping the ITAE under **2.60**. This is due to the nonlinear derivative gain component $\gamma(de/dt)$, which automatically scales down to zero when derivative rate-of-change spikes are detected, filtering out the noise.

### 6.3 Out-of-Distribution Extrapolation Test ($J = 0.03$)

To test the generalization limit, we simulated a system with inertia $J = 0.03\text{ kg}\cdot\text{m}^2$ (50% larger than the training limit of $J = 0.02$) under a step load torque disturbance of $0.02\text{ N}\cdot\text{m}$ at $t = 1.5\text{s}$.

| Controller | Rise Time (s) | Overshoot (%) | Settling Time (s) | IAE | ISE | ITAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L-PID (ZN)** | 0.2090 | 2.56% | 0.3610 | 3.8094 | 57.4793 | 0.4347 |
| **N-PID (ZN)** | 0.2070 | 12.54% | 3.9990 | 5.7730 | 60.5481 | 3.2946 |
| **N-PID (PSO)**| 0.2090 | 10.83% | 1.2490 | 5.0541 | 60.2010 | 1.3410 |
| **N-PID (ML)** | 0.2090 | 10.84% | 1.3670 | 5.0941 | 60.2152 | 1.4124 |

*Analysis:* Under extrapolation conditions with load disturbances, the ML-predicted N-PID controller remains highly stable and performs nearly identically to the PSO-optimal controller (ITAE of 1.4124 vs 1.3410). This indicates that the Random Forest model generalized effectively outside the boundary envelope.

---

## 7. Conclusions & Future Work

### 7.1 Key Findings
* **N-PID Noise Filtering:** The N-PID structure offers exceptional robustness against measurement noise due to the nonlinear derivative factor $\gamma(de/dt)$, whereas standard L-PID becomes highly oscillatory.
* **Tuning Efficiency:** Online PSO-tuning achieves optimal controller damping (85% reduction in ITAE) but is too slow for real-time adjustments ($30.1\text{s}$).
* **ML Integration:** Offline training of a Random Forest Regressor allows for near-instant ($15.9\text{ms}$) gain predictions. The predicted gains maintain 100% control stability and achieve an ITAE within 1.6% of the true PSO-optimal baseline.

### 7.2 Future Work: Online Adaptive Reinforcement Learning
While the offline supervised learning approach successfully maps physical parameters to gains, it relies on:
1. Having an accurate measurement of the system parameters ($J, B, T_c$) beforehand.
2. Assuming the system parameters are constant over time.

To overcome these, a natural next step is implementing **Online Reinforcement Learning (RL)**, such as Deep Deterministic Policy Gradient (DDPG) or Twin Delayed DDPG (TD3). An RL agent can:
* Treat the PID gains as continuous action outputs.
* Learn directly from tracking error states and reward functions (e.g., negative ITAE) in real-time.
* Dynamically adjust gains in response to unmodeled parameter drift (such as motor heating increasing armature resistance $R$ or load variations) without requiring explicit parameter estimation, enabling fully adaptive online control.
