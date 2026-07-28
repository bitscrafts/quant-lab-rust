//! Core traits for quantitative finance components.
//!
//! This module defines traits that enable composable, extensible
//! quantitative finance workflows. All traits are designed to be
//! object-safe where feasible, allowing for runtime polymorphism.
//!
//! # Organization
//!
//! - [`cv`]: Cross-validation trait
//! - [`labeler`]: Event labeling trait
//! - [`sizer`]: Bet sizing trait
//! - [`weighter`]: Sample weighting trait
//! - [`process`]: Stochastic process simulation trait
//! - [`pricer`]: Option pricing and Greeks traits
//! - [`factor`]: Factor model trait
//! - [`microstructure`]: Market microstructure traits
//! - [`detector`]: Structural break detection trait

pub mod cv;
pub mod detector;
pub mod factor;
pub mod labeler;
pub mod microstructure;
pub mod pricer;
pub mod process;
pub mod sizer;
pub mod weighter;

pub use cv::CrossValidator;
pub use detector::{StructuralBreak, StructuralBreakDetector};
pub use factor::FactorModel;
pub use labeler::Labeler;
pub use microstructure::{ImpactModel, OrderBookOps};
pub use pricer::{Greeks, OptionPricer, OptionType};
pub use process::StochasticProcess;
pub use sizer::BetSizer;
pub use weighter::SampleWeighter;
