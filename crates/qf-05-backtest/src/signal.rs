//! Trading signals and position states.
//!
//! See `README.md` in this directory for the module overview.

/// A trading signal emitted by a strategy for a single bar.
///
/// Signals are deterministic instructions: the engine converts them into
/// position transitions, applying transaction costs on every state change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Enter a long position (or add to an existing one, depending on engine
    /// policy).
    Buy,
    /// Exit a long position. In Phase 5 we are long-only, so a `Sell` means
    /// "go to flat".
    Sell,
    /// Do nothing this bar.
    Hold,
}

/// The position the engine currently holds.
///
/// Phase 5 is long-only: `Short` is declared here so future phases can extend
/// the engine without breaking the public API, but the engine rejects
/// `allow_short = false` configurations that would produce a short position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// Holding the asset.
    Long,
    /// Short selling the asset (reserved for future phases).
    Short,
    /// No position.
    Flat,
}

impl Position {
    /// Returns true when the position is `Long` or `Short` (i.e. exposed to
    /// the market).
    pub fn is_exposed(self) -> bool {
        matches!(self, Position::Long | Position::Short)
    }

    /// Returns true when the position is `Flat`.
    pub fn is_flat(self) -> bool {
        matches!(self, Position::Flat)
    }
}