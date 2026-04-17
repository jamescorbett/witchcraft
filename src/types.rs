//! Core types for the witchcraft library.
//!
//! Re-exports key types from submodules for convenience.

pub use crate::filter::{Filter, FilterCondition, FilterLogic, FilterOp, MetadataValue};
pub use crate::schema::{MetadataField, MetadataFieldType, MetadataSchema};

/// A search result returned by [`Witchcraft::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Relevance score (higher is better).
    pub score: f32,
    /// Metadata column values for this document.
    pub metadata: std::collections::HashMap<String, MetadataValue>,
    /// Body text split into sub-chunks.
    pub bodies: Vec<String>,
    /// Index of the best-matching sub-chunk.
    pub matched_sub_idx: u32,
    /// Date string (ISO 8601).
    pub date: String,
}

/// A document pointer combining document row identity and sub-chunk index.
/// Used internally for score fusion between search methods.
pub type DocPtr = (String, u32); // (doc_uuid, sub_idx)
