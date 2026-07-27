# src/ — quant-microstructure source

[← back to crate README](../README.md)

## Module Map

| File | Purpose | Key types / functions |
|---|---|---|
| `lib.rs` | Crate root, re-exports | `pub use` of all public APIs |
| `error.rs` | Error enum | `MicroError` (OrderNotFound, InvalidOrder, EmptyBook, InsufficientLiquidity, InvalidTickSize, DimensionMismatch, InsufficientData) |
| `types.rs` | Core data types | `Side` (Bid/Ask), `Order`, `Level`, `Fill`, `Trade`, `BookSnapshot` |
| `orderbook.rs` | The limit order book | `OrderBook` with `BTreeMap<u64, Vec<Order>>` per side plus `HashMap` id index; price-time priority |
| `flow.rs` | Flow metrics | `order_flow_imbalance` (OFI), `vwap`, `trade_imbalance` |
| `impact.rs` | Market impact | `sqrt_impact` (Almgren-Chriss), `linear_impact`, `execution_cost`, `ExecutionCost` |

## Architecture

```
OrderBook
  bids: BTreeMap<u64, Vec<Order>>   <- highest key = best bid
  asks: BTreeMap<u64, Vec<Order>>   <- lowest  key = best ask
  tick_size: u64
  index: HashMap<u64, (Side, u64)>  <- O(1) cancel lookup by id
```

Operations:
- `add_order`: O(log L + 1) amortised --- insert into BTreeMap and push
  to the level's vector; update the id index.
- `cancel_order`: O(1) for index lookup + O(K) to splice the order
  out of its level vector (K = orders at that level).
- `market_order`: O(L * K) in the worst case --- walks the opposite
  side level by level, consuming the front of each vector.

## Testing

- Unit tests in each module file (`#[cfg(test)] mod tests`)
- Integration contract in `tests/micro_tests.rs` (15 tests)
- 42 tests total, all passing; `cargo clippy -D warnings` clean

## Design Decisions

### Why integer tick prices?

Floating-point comparison is non-associative and rounding-dependent.
The order book must compare prices exactly (an order either matches
the tick grid or it does not). `u64` ticks make this trivial and
unambiguous.

### Why BTreeMap?

A `BTreeMap<u64, _>` provides ordered price levels with O(log L)
insert, delete and best-price lookup. A `HashMap` would not preserve
order; a sorted `Vec` would shift data on every insert. The price is
the key, so the book never has to sort --- ordering is structural.

### Why a separate id index?

Without the `HashMap<id, (side, price)>` index, cancelling by id
would require scanning every price level and every order. The index
trades a small memory cost for O(1) lookup of the order's location.

### Why hand-rolled?

Production trading systems use heavily optimised LOB implementations
(lock-free queues, LMAX Disruptor patterns, SIMD order matching).
This crate is a teaching artefact: every line is short enough to read
in one sitting, and the price-time priority rule is visible in the
code rather than hidden behind a library API.