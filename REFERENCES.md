## Random Number Generation (PCG)

O’Neill, M. E. (2014).  
_PCG: A Family of Simple Fast Space-Efficient Statistically Good Random Number Generators._  
HMC CS Technical Report 2014-0905.  
http://www.pcg-random.org/

Describes the PCG32 RNG used as the default random number generator.

---

## Effective Sample Size (ESS)

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).  
_Bayesian Data Analysis_ (3rd ed.). CRC Press.

Explains the classical truncated-autocorrelation ESS estimator implemented here.

---

## Convergence Diagnostics (R-hat)

Gelman, A., & Rubin, D. B. (1992).
_Inference from Iterative Simulation Using Multiple Sequences._ Statistical Science, 7(4), 457–472.

Introduces the Gelman-Rubin convergence diagnostic (R-hat) and the concept of overdispersed initialization for multiple chains.

---

## Chain Initialization

Stan Development Team (2024).
_Stan Reference Manual - Initialization._

https://mc-stan.org/docs/reference-manual/initialization.html

Describes Stan's default initialization strategy uses Uniform(-2, 2) on the unconstrained parameter scale. This library follows the same approach for generating dispersed starting points when users don't provide custom initial values.

---

## Metropolis–Hastings

Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953).
_Equation of State Calculations by Fast Computing Machines._ Journal of Chemical Physics, 21(6), 1087–1092.

Hastings, W. K. (1970).
_Monte Carlo Sampling Methods Using Markov Chains and Their Applications._ Biometrika, 57(1), 97–109.

These papers introduce the Metropolis algorithm and Hastings' generalization that underpin the random-walk MH sampler implemented here.

---

## Adaptive MCMC

Andrieu, C. & Thoms, J. (2008).
_A Tutorial on Adaptive MCMC._ Statistics and Computing, 18(4), 343–373.
https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf

Comprehensive tutorial on adaptive MCMC methods, including conditions for ergodicity of adaptive chains. The Robbins-Monro step size adaptation implemented here is covered in Section 3.
