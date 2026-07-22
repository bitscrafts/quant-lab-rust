# Table of Contents — Learning Quant Finance in Rust

[← back to README](../README.md)

This table tracks book chapters, their spec phases, and companion crates. The
book uses the kaobook LaTeX template with TikZ figures.

## Build Instructions

### Prerequisites

The kaobook class files (`kaobook.cls`, `kao.sty`, etc.) are included in the
`book/` directory. Original source: https://github.com/fmarotta/kaobook

### First-Time Setup

Create remote directory structure:
```bash
ssh mvcorrea@lnx "mkdir -p ~/podman/latex-remote/quant-finance/{src,build}"
```

Start the LaTeX container if not running:
```bash
ssh mvcorrea@lnx "podman start latex-compiler-shared"
```

### Compilation Steps

```bash
# 1. Sync all book files (including kaobook classes)
rsync -avz --delete docs/quant-finance/book/ \
    mvcorrea@lnx:~/podman/latex-remote/quant-finance/src/

# 2. Compile with biber for bibliography
ssh mvcorrea@lnx "podman exec latex-compiler-shared sh -c '
    cd /workspace/quant-finance/src && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    biber --output-directory ../build main && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex
'"

# 3. Download compiled PDF
scp mvcorrea@lnx:~/podman/latex-remote/quant-finance/build/main.pdf \
    docs/quant-finance/book/build/quant-finance-book.pdf
```

### Quick Compile (single command)

```bash
rsync -avz --delete docs/quant-finance/book/ mvcorrea@lnx:~/podman/latex-remote/quant-finance/src/ && \
ssh mvcorrea@lnx "podman exec latex-compiler-shared sh -c '
    cd /workspace/quant-finance/src && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    biber --output-directory ../build main && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex
'" && \
scp mvcorrea@lnx:~/podman/latex-remote/quant-finance/build/main.pdf \
    docs/quant-finance/book/build/quant-finance-book.pdf
```

### Known Issues

1. **TikZ with pgfplots in kaobook**: Complex pgfplots figures can cause
   "parameter stack size exceeded" errors. Use simpler TikZ primitives or
   externalize figures.

2. **Empty bibliography warning**: Expected until references.bib has entries
   actually cited in the text.

## Kaobook Template

The book uses kaobook with margin notes for formulas and insights. Class files
are bundled in the book directory:

- `kaobook.cls` — Main document class
- `kao.sty` — Core kaobook utilities
- `kaobiblio.sty` — Bibliography styling
- `kaorefs.sty` — Cross-reference utilities
- `kaotheorems.sty` — Theorem environments

---

## Part I: Foundations

| Ch | Title | Spec | Crate | LaTeX | Status |
|----|-------|------|-------|-------|--------|
| 01 | Hello Finance: Credit Card Fraud | [Phase 1](../spec.md#phase-1-hello-finance--credit-card-fraud-detection) | `qf-01-fraud` | `chapters/ch01.tex` | **done** |
| 02 | Risk Basics: Loan Default | [Phase 2](../spec.md#phase-2-risk-basics--loan-default-prediction) | `qf-02-loan` | `chapters/ch02.tex` | **done** |
| 03 | Market Data: Stock Prices | [Phase 3](../spec.md#phase-3-market-data--stock-price-analysis) | `qf-03-stocks` | `chapters/ch03.tex` | **done** |

## Part II: Core Concepts

| Ch | Title | Spec | Crate | LaTeX | Status |
|----|-------|------|-------|-------|--------|
| 04 | Returns and Volatility | [Phase 4](../spec.md#phase-4-returns--volatility) | `qf-04-returns` | `chapters/ch04.tex` | planned |
| 05 | Your First Backtest | [Phase 5](../spec.md#phase-5-first-backtest) | `qf-05-backtest` | `chapters/ch05.tex` | planned |

## Part III: Quantitative Methods

| Ch | Title | Spec | Crate | LaTeX | Status |
|----|-------|------|-------|-------|--------|
| 06 | Foundations: Moments and Fat Tails | [Phase 6](../spec.md#phase-6-foundations--moments-and-fat-tails) | `quant-core` | `chapters/ch06.tex` | planned |
| 07 | Time Series: Stationarity | [Phase 7](../spec.md#phase-7-time-series--stationarity-and-fractional-differentiation) | `quant-timeseries` | `chapters/ch07.tex` | planned |
| 08 | Volatility Models | Phase 8 | `quant-vol` | `chapters/ch08.tex` | planned |
| 09 | Stochastic Processes | Phase 9 | `quant-stochastic` | `chapters/ch09.tex` | planned |

## Part IV: Derivatives and Portfolio

| Ch | Title | Spec | Crate | LaTeX | Status |
|----|-------|------|-------|-------|--------|
| 10 | Options Pricing | Phase 10 | `quant-options` | `chapters/ch10.tex` | planned |
| 11 | Portfolio Optimization | Phase 11 | `quant-portfolio` | `chapters/ch11.tex` | planned |
| 12 | Factor Models | Phase 12 | `quant-factors` | `chapters/ch12.tex` | planned |

## Part V: Advanced Topics

| Ch | Title | Spec | Crate | LaTeX | Status |
|----|-------|------|-------|-------|--------|
| 13 | Market Microstructure | Phase 13 | `quant-microstructure` | `chapters/ch13.tex` | planned |
| 14 | AFML: From Research to Production | Phase 14 | `quant-backtest` | `chapters/ch14.tex` | planned |

## Appendices

| App | Title | LaTeX | Status |
|-----|-------|-------|--------|
| A | Mathematical Notation | `chapters/appA.tex` | planned |
| B | Rust Patterns for Finance | `chapters/appB.tex` | planned |
| C | Dataset Sources | `chapters/appC.tex` | planned |

---

## Figures (TikZ)

Figures live in `figures/` and are included via `\input{figures/filename}`:

| Figure | Chapter | Description | Status |
|--------|---------|-------------|--------|
| `confusion-matrix.tex` | 01 | Confusion matrix visualization | planned |
| `roc-curve.tex` | 02 | ROC curve and AUC | planned |
| `candlestick.tex` | 03 | OHLCV candlestick chart | planned |
| `returns-dist.tex` | 04 | Normal vs fat-tailed returns | planned |
| `sma-crossover.tex` | 05 | SMA crossover signals | planned |
| `gbm-paths.tex` | 06 | GBM sample paths | planned |
| `fat-tails.tex` | 06 | Kurtosis comparison | planned |
| `acf-plot.tex` | 07 | Autocorrelation function | planned |
| `adf-test.tex` | 07 | Stationarity visualization | planned |
| `ffd-tradeoff.tex` | 07 | Memory vs stationarity | planned |
| `garch-vol.tex` | 08 | Volatility clustering | planned |
| `brownian.tex` | 09 | Brownian motion paths | planned |
| `bs-surface.tex` | 10 | Black-Scholes surface | planned |
| `greeks.tex` | 10 | Greeks visualization | planned |
| `efficient-frontier.tex` | 11 | Mean-variance frontier | planned |
| `pca-variance.tex` | 12 | PCA explained variance | planned |
| `lob.tex` | 13 | Limit order book | planned |
| `triple-barrier.tex` | 14 | Triple-barrier labeling | planned |

---

## Directory Structure

```
book/
├── toc.md                  # This file (metadata/index)
├── main.tex                # Main LaTeX document (kaobook)
├── references.bib          # Bibliography
├── chapters/               # Chapter .tex files
│   ├── ch01.tex           # Hello Finance
│   ├── ch02.tex           # Risk Basics
│   ├── ...
│   ├── ch14.tex           # AFML
│   ├── appA.tex           # Math notation appendix
│   ├── appB.tex           # Rust patterns appendix
│   └── appC.tex           # Datasets appendix
├── figures/                # TikZ source files
│   ├── confusion-matrix.tex
│   ├── roc-curve.tex
│   └── ...
├── images/                 # Raster images (if any)
└── build/                  # Compiled output (gitignored)
    └── main.pdf
```

---

## Workflow

1. **Implement crate** — Complete spec phase, all tests pass
2. **Write chapter** — Document implementation journey in LaTeX
3. **Create figures** — TikZ diagrams for key concepts
4. **Update toc.md** — Mark chapter status as `done`
5. **Compile book** — Run latex-remote skill
6. **Review PDF** — Check output, iterate if needed

---

## Chapter Template

Each chapter file (`chapters/chNN.tex`) follows this structure:

```latex
% Chapter NN: Title
% Companion code: crates/<crate-name>

\begin{objectives}
\item Objective 1
\item Objective 2
\item Objective 3
\end{objectives}

\section{Introduction}

\section{Mathematical Foundation}

\formulamargem{Key Formula}{$formula$}

\section{Implementation in Rust}

\begin{companioncode}{<crate-name>}
See \texttt{crates/<crate-name>/}
\end{companioncode}

\begin{lstlisting}
// Rust code here
\end{lstlisting}

\section{Results and Analysis}

\begin{keyinsight}
Key insight from implementation.
\end{keyinsight}

\section{Exercises}

\section{Summary}
```
