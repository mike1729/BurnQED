//! Parquet I/O for search trajectory data.
//!
//! Provides types for recording proof search states and reading/writing
//! them as Parquet files for EBM training.

pub mod reader;
pub mod types;
pub mod writer;

pub use reader::TrajectoryReader;
pub use types::{
    SearchResult, SearchStats, TheoremIndex, TheoremTask, TrajectoryLabel, TrajectoryRecord,
    TrajectorySummary,
};
pub use writer::TrajectoryWriter;
