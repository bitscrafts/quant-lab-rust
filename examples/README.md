# Examples

This directory contains symlinks to the actual examples in
[`crates/quant-lib/examples/`](../crates/quant-lib/examples/).

## Structure

```
examples/
├── portfolio/  -> crates/quant-lib/examples/portfolio/   (25 Wall Street implementations)
├── projects/   -> crates/quant-lib/examples/projects/    (5 real-world projects)
└── solutions/  -> crates/quant-lib/examples/solutions/   (14 chapter solutions)
```

## Running Examples

All examples are registered under the `quant-lib` package:

```bash
# Core examples (01-16)
cargo run -p quant-lib --example 01_mean_variance

# Real-world projects
cargo run -p quant-lib --example projects-01_momentum

# Chapter solutions
cargo run -p quant-lib --example solutions-ch01_fraud_solutions

# Portfolio implementations (25 Wall Street examples)
cargo run -p quant-lib --example portfolio-01_monte_carlo_foundations
```

## Portfolio Examples (NEW)

25 implementations based on Zhou's "Practical Guide to Quantitative Finance Interviews":

| Category | Examples | Status |
|----------|----------|--------|
| Probability (01-05) | Monte Carlo, Bayesian, Hypothesis Testing | TO BE IMPLEMENTED |
| Stochastic (06-10) | Markov Chains, Brownian, Ito, Jump-Diffusion | TO BE IMPLEMENTED |
| Options (11-15) | Black-Scholes, Binomial, Greeks, Exotics | TO BE IMPLEMENTED |
| Portfolio (16-20) | Markowitz, Black-Litterman, Risk Parity | TO BE IMPLEMENTED |
| ML (21-25) | HMM Regimes, Neural Pricing, RL Trading | TO BE IMPLEMENTED |

See [`portfolio/README.md`](portfolio/README.md) for details.

## Full Documentation

See [`crates/quant-lib/examples/README.md`](../crates/quant-lib/examples/README.md)
for complete documentation of all examples.
