# Examples

This directory contains symlinks to the actual examples in
[`crates/quant-lib/examples/`](../crates/quant-lib/examples/).

## Structure

```
examples/
├── projects/   -> crates/quant-lib/examples/projects/
└── solutions/  -> crates/quant-lib/examples/solutions/
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
```

## Full Documentation

See [`crates/quant-lib/examples/README.md`](../crates/quant-lib/examples/README.md)
for complete documentation of all examples.
