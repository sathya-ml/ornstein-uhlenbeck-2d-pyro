# 2D Ornstein-Uhlenbeck Process Variational Inference in Pyro-PPL

The Ornstein-Uhlenbeck (OU) process is a stochastic process widely used in physics, finance, and other fields. It exhibits mean-reverting behavior, consistently moving towards a central value or long-term average over time. This process can be extended to multiple dimensions, known as the multi-dimensional Ornstein-Uhlenbeck process. The OU process is characterized by its stationarity, meaning its mean and variance remain constant over time. Additionally, it is a Gaussian process, which implies that all its finite-dimensional distributions follow a Gaussian distribution. As a continuous-time Markov process, it is governed by the following stochastic differential equation:

```math
dx_t = -\beta x_t dt + \sigma dW_t
```

where:

- $x_t$ is a 2-dimensional vector representing the position at time $t$
- $\beta$ is a constant 2x2 drift matrix
- $\sigma$ is a constant 2x2 diffusion matrix
- $W_t$ is a 2-dimensional Wiener process (standard Brownian motion)

In two dimensions, the OU process has diverse applications, including:

- Modeling the velocity of a particle in a fluid
- Simulating eye movement patterns in gaze modeling
- Describing financial market fluctuations
- Analyzing animal movement patterns in ecology
- Representing wind velocity changes in meteorology
- Modeling neuronal firing rates in neuroscience

More generally, it can describe any mean-reverting motion in complex systems across various scientific disciplines.

This repository contains a Pyro-PPL implementation of the Ornstein-Uhlenbeck process in two dimensions, with inference performed over the drift ($\beta$) and diffusion ($\sigma$) matrices. Inference over stochastic processes is usually performed using MCMC techniques, which provide guarantees on the estimation accuracy, but are generally speaking slow. Pyro allows for performing stochastic variational inference (SVI) on the model, which is faster, but SVI lacks guarantees on the estimation accuracy. We assume that the data is generated from a stationary Ornstein-Uhlenbeck process, which means that the drift matrix $\beta$ has to be positive definite. We further assume that the data consist of N independent simulations of equal length, each from the same process with the same initial conditions. We also assume that the drift term, which is sometimes included in the above equation, equals zero.

Using SVI might introduce some error in the estimation of the drift and diffusion matrices, but has the advantage of shorter inference time, especially for larger datasets. This makes it suitable for conditions where there is plenty of data, and a "quick" estimate of model parameters is more valuable than an "exact" posterior.

## Installation, Usage and Features

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

The [src/](src/) directory contains two files. The [src/datagen.py](src/datagen.py) file contains code to generate synthetic data from a stationary Ornstein-Uhlenbeck process. The [src/ornstein_uhlenbeck.py](src/ornstein_uhlenbeck.py) file contains the Pyro model, an inference function, and example code that generates synthetic data from `datagen.py`, performs inference on the model using SVI and compares the actual and inferred values to gauge the inference accuracy. It will also plot the loss trace.

To run `ornstein_uhlenbeck.py` run:

```bash
python src/ornstein_uhlenbeck.py
```

Alternatively, you can import the model and inference code into your own project.

### The Model

The model defines the following generative process:

1. Sample the drift matrix $\beta$ from a Wishart distribution, given the degrees of freedom and scale matrix. This ensures that the drift matrix is positive definite, which guarantees stationarity.
2. Sample the correlation matrix $\Sigma$ from an LKJ distribution, given the concentration parameter.
3. Sample the standard deviations from a Half-Cauchy distribution, given the scale parameter.
4. Transform the LKJ-distributed correlation matrix into covariance matrix $\Sigma$, using the standard deviations.
5. Set the drift term and the initial value of the process $x_0$ to zero.

Using the above, we proceed to sample from the model by using the Euler-Maruyama method combined with Pyro's primitives and functions.

### Inference

Inference is performed using SVI, using the AutoDelta guide, which is effectively a set of Delta distributions over the latent variables in the model. They correspond to a MAP estimate of the posterior. To ensure decent convergence and avoid overfitting, we perform early stopping based on the validation loss.
A hardcoded rule governs when to stop: if after at least 30 steps the difference between the maximum loss in the last 10 steps and the minimum loss in the last 10 steps is less than 2.0, we stop the inference procedure and return the current estimate of the posterior.

Using the Wishart prior on the diffusion matrix effectively constrains inference to only SVI techniques. Using MCMC will fail due to issues in Pyro's MCMC implementation related to the Wishart distribution and its positivity constraints (see [here](https://forum.pyro.ai/t/mcmc-for-mixture-of-gaussians/5144)).
If you want to perform MCMC inference, a good alternative to consider is to use a LKJ prior on the drift matrix.

### Performance

The model performs well at learning the drift matrix, but sometimes struggles with getting a good estimate of the diffusion matrix. This is most likely due to the fact that the diffusion matrix is not as well constrained by the data. This might be related to the identifiability problem, with limited data there are multiple matrices that explain the data equally well.

The Wishart prior on the drift matrix was chosen because empirically it worked better with SVI than other priors that enforce positive definiteness.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
