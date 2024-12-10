# Planning and Learning in Risk-Aware Restless Multi-Arm Bandit Problem

This project implements simulations for planning and learning in the context of risk-aware and risk-neutral Restless Multi-Armed Bandit (RMAB) problems. 

## Project Structure

The project includes two main simulation modules and several supporting scripts:

### Main Modules
- **`main_planning.py`**  
  Runs simulations for **planning** under various parameters to compute Whittle indices and evaluate planning strategies.

- **`main_learning.py`**  
  Executes simulations for **learning** to evaluate how well agents learn Whittle indices and adapt under different settings.

### Supporting Scripts
- **`whittle.py`**  
  Contains classes and methods for computing Whittle indices:
  - `Whittle`: For risk-neutral index computation.
  - `RiskAwareWhittle`: Extends Whittle computation with risk-aware considerations.
  Implements binary search methods in addition to backward induction for index computation.

- **`Markov.py`**  
  Defines reward structures and dynamics for arms:
  - Linear and non-linear reward models.
  - Transition dynamics for Markov decision processes for either the structured or real-world models.

- **`processes.py`**  
  Core logic for running multi-episode simulations under finite time horizons:
  - Simulates various policies and processes.
  - Computes the risk-aware objective for evaluating strategies.

- **`learning.py`**  
  Implements processes to learn Whittle indices over multiple iterations:
  - Tracks errors and the regret values.
  - Updates dynamics and policies based on learning outcomes.

- **`utils.py`**  
  Utility functions for:
  - Managing and running multiple parameter combinations.
  - Saving and visualizing results.
  - Plotting performance metrics like regret.

- **`histogram_statistics.py`**  
  This script generates and plots histograms for the reported statistics in the paper. It computes metrics such as mean, min, max, and the portion of values below zero, providing insights into relative improvements and other key measures.

- **`plots_utility.py`**  
  Generates visualizations for utility functions discussed in the paper. The script plots various utility representations, including risk-averse and risk-neutral utility functions, based on total rewards.


## Running Simulations

1. **Planning Simulations**
    ```bash
    python main_planning.py
    ```
    The results will be saved in a folder named as 'output-finite'.
    The following commands would generate the statistics and the histogram
    ```bash
    python parameter_change_scores.py
    ```
    ```bash
    python plots_utility.py
    ```

2. **Learning Simulations**
    ```bash
    python main_learning.py
    ```
   The results/plots will be saved in a folder named as 'output-learn-finite'.

### Dependencies

This project requires Python 3.x and the following Python libraries:
- `numpy`
- `scipy`
- `joblib`
- `matplotlib`
- `pandas`
- `openpyxl`

To install these dependencies, run:
```bash
pip install numpy scipy joblib matplotlib pandas openpyxl
```

### Results

Results of the simulation for planning is saved in `./planning-finite/`, and the results pf simulation
for learning is saved in `./learning-finite-.../`.