# Evaluation of Gradient Descent, Genetic Algorithm, and Differential Evolution for Global Optimization

**Author:** Tshepiso Clearance Mahoko  
**Institution:** University of Johannesburg  
**Student Number:** 220015607  

---

## Abstract
Optimization is fundamental in science and engineering, where the objective is to identify optimal solutions under given constraints. Traditional methods such as Gradient Descent (GD) are effective for convex problems but struggle in complex, multi-modal landscapes. Evolutionary Algorithms (EAs), including Genetic Algorithm (GA) and Differential Evolution (DE), offer population-based global search capabilities without requiring gradient information.  

This project presents a comparative evaluation of GD, GA, and DE on benchmark optimization functions in multi-dimensional search spaces, analysing convergence behaviour, robustness, and success rates in locating global minima.

---

## 1. Introduction
Optimization problems frequently arise in engineering design, machine learning, and control systems. Gradient Descent iteratively moves toward a minimum using gradient information, making it efficient for convex problems but vulnerable to local minima in non-convex landscapes.  

Evolutionary Algorithms, inspired by natural selection, explore the search space globally using populations of candidate solutions. This study compares GD, GA, and DE to assess their effectiveness on complex, multi-modal benchmark functions.

---

## 2. Problem Statement
Finding global minima in high-dimensional, multi-modal functions is challenging due to the presence of numerous local minima and deceptive landscapes. Gradient-based methods often converge prematurely, while evolutionary methods offer global exploration but vary in efficiency.  

This study evaluates GD, GA, and DE to determine which method most reliably identifies global optima across diverse optimization landscapes.

---

## 3. Optimization Algorithms

### 3.1 Gradient Descent (GD)
Gradient Descent is a first-order optimization algorithm that updates solutions using gradient information:

\[
x_{k+1} = x_k - \eta \nabla f(x_k)
\]

While computationally efficient, GD is sensitive to initialization, learning rate selection, and local minima, limiting its reliability in non-convex problems.

---

### 3.2 Genetic Algorithm (GA)
Genetic Algorithms are population-based stochastic optimizers inspired by biological evolution. Core operations include:
- Initialization
- Selection
- Crossover
- Mutation  

GA performs global search and avoids reliance on gradient information, making it suitable for complex and noisy optimization landscapes.

---

### 3.3 Differential Evolution (DE)
Differential Evolution is a population-based algorithm designed for continuous global optimization. New candidate solutions are generated using weighted differences between population members:

\[
v_i = x_r + F \cdot (x_s - x_t)
\]

DE balances exploration and exploitation effectively and typically converges faster than GA with fewer control parameters.

---

## 4. Benchmark Functions
The following benchmark functions were selected to represent diverse optimization challenges:

| Function | Characteristics |
|--------|----------------|
| Rastrigin | Highly multi-modal with many local minima |
| Ackley | Flat outer regions and a central basin |
| Schwefel | Deceptive local minima |
| Six-Hump Camel | Multiple local minima with two global minima |

All functions were evaluated in n-dimensional search spaces (n ≥ 2).

---

## 5. Methodology
1. **Initialization:** Random initialization within function bounds  
2. **Parameters:**  
   - GD: learning rate, max iterations  
   - GA: population size, mutation rate, generations  
   - DE: mutation factor (F), crossover rate (CR), generations  
3. **Evaluation Metrics:**  
   - Best fitness value  
   - Convergence rate  
   - Success rate  
4. **Visualization:**  
   - Convergence curves  
   - 3D surface plots  

---

## 6. Experimental Results

### Key Observations
- **Gradient Descent:** Frequently trapped in local minima with high variability  
- **Genetic Algorithm:** Moderate performance with improved robustness over GD  
- **Differential Evolution:** Consistently achieved the lowest fitness values and highest success rates  

DE achieved success rates between **86% and 100%** across all benchmark functions, outperforming both GD and GA.

---

## 7. Discussion
Results highlight the limitations of gradient-based optimization in complex landscapes. Population-based evolutionary algorithms demonstrated superior robustness and scalability. Differential Evolution showed the best balance between convergence speed and global exploration, making it highly effective for high-dimensional, multi-modal optimization problems.

---

## 8. Conclusion
This study confirms that Evolutionary Algorithms, particularly Differential Evolution, are more reliable than Gradient Descent for global optimization in complex search spaces. DE consistently outperformed GD and GA across all benchmarks, reinforcing its suitability for real-world optimization problems involving non-convex and high-dimensional landscapes.

Future work may explore hybrid and adaptive evolutionary strategies for large-scale optimization problems.

---

## References
1. Ruder, S. (2016). *An overview of gradient descent optimization algorithms*. arXiv:1609.04747.  
2. Katoch, S., Chauhan, S. S., & Kumar, V. (2021). *A review on genetic algorithm: past, present, and future*. Multimedia Tools and Applications.  
3. Storn, R., & Price, K. (1997). *Differential evolution – a simple and efficient heuristic for global optimization*. Journal of Global Optimization.  
4. Liu, Y., et al. (2019). *Benchmark functions for evaluating evolutionary algorithms*. Applied Soft Computing.  

---
