# quant-microstructure

Market microstructure: the limit order book, order flow imbalance,
and market impact models. Phase 13 crate of the quant-finance
curriculum.

[← back to quant-lab](../README.md)

## Overview

`quant-microstructure` builds the intraday machinery that the
portfolio theory of `quant-portfolio` and the factor attribution of
`quant-factors` abstract away: the limit order book (LOB) with
price-time priority, the order flow imbalance (OFI) signal, and the
square-root (Almgren-Chriss) market impact law.

The order book uses `BTreeMap<u64, Vec<Order>>` keyed by price level
with integer (tick) prices to avoid floating-point comparison issues.
Orders at the same price execute in FIFO (price-time) priority. All
data structures are hand-rolled --- no external order book libraries.

## Modules

| Module | Public API |
|---|---|
| `types` | `Side`, `Order`, `Level`, `Fill`, `Trade`, `BookSnapshot` |
| `orderbook` | `OrderBook` with `add_order`, `cancel_order`, `market_order`, `best_bid`, `best_ask`, `mid_price`, `spread`, `depth`, `total_volume`, `order_count` |
| `flow` | `order_flow_imbalance`, `vwap`, `trade_imbalance` |
| `impact` | `sqrt_impact`, `linear_impact`, `execution_cost`, `ExecutionCost` |
| `error` | `MicroError` |

## Dependencies

- `quant-core` --- shared error and numeric traits
- `thiserror` --- derive `Error`

Dev dependencies: `approx`, `quant-core`.

## Usage

```rust
use quant_microstructure::{OrderBook, Side, Order};

let mut book = OrderBook::new(1);
book.add_order(Order { id: 1, side: Side::Bid, price: 100, quantity: 10, timestamp: 1 }).unwrap();
book.add_order(Order { id: 2, side: Side::Ask, price: 101, quantity: 20, timestamp: 2 }).unwrap();
assert_eq!(book.spread(), Some(1));
let fills = book.market_order(Side::Bid, 15);
assert_eq!(fills.len(), 1);
```

## Examples

- `lob_demo` --- build a book, execute a market buy, inspect fills and post-trade state
- `ofi_demo` --- OFI series, trade imbalance, and execution-cost estimation

## Tests

- 26 unit tests across `types`, `orderbook`, `flow`, `impact`
- 15 integration tests in `tests/micro_tests.rs` (TDD contract)
- 1 doc test
- Total: 42 tests, all passing; clippy clean

## Design Notes

- **Integer tick prices**: prices are `u64` ticks, so tick-size validation
  and price comparison are exact, not subject to float rounding.
- **BTreeMap per side**: ordered price levels come for free; the best
  bid is `iter().next_back()`, the best ask is `iter().next()`.
- **HashMap id index**: O(1) lookup of `(side, price)` by order id
  makes cancel fast.
- **Price-time priority**: at each price level, orders are stored in a
  `Vec` in insertion (FIFO) order; market orders consume the front
  first.

## Related Crates

- [`quant-portfolio`](../quant-portfolio/): Markowitz frontier (the
  frictionless world this crate adds friction to)
- [`quant-timeseries`](../quant-timeseries/): OLS regression used for
  factor exposure estimation
- [`quant-factors`](../quant-factors/): PCA and Fama-French

## References

- Cont, R., Kukanov, A., Stoikov, S. (2014). _The price impact of
  order book events_. Journal of Financial Econometrics.
- Almgren, R., Chriss, N. (2000). _Optimal execution of portfolio
  transactions_. Journal of Risk.
- Foucault, T., Pagano, M., Roell, A. (2005). _Market Liquidity:
  Theory, Evidence, and Policy_. Oxford University Press.