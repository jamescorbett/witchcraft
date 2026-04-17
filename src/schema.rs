//! Metadata schema definition and Arrow schema construction for LanceDB tables.
//!
//! Users define a [`MetadataSchema`] describing their metadata columns at table
//! creation time. The library combines these with the fixed document/embedding
//! columns to produce the full Arrow schema used by LanceDB.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use crate::EMBEDDING_DIM;

/// Supported metadata column types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetadataFieldType {
    String,
    Number,
    Bool,
}

/// A single metadata column definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataField {
    pub name: String,
    pub field_type: MetadataFieldType,
    /// If true, a scalar index will be created on this column for fast filtering.
    pub indexed: bool,
}

/// Schema description for user-defined metadata columns.
///
/// Passed to [`Witchcraft::new`] to define the table schema.
/// An empty schema is valid (no metadata columns).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataSchema {
    pub fields: Vec<MetadataField>,
}

impl MetadataSchema {
    pub fn new() -> Self {
        Self { fields: vec![] }
    }

    /// Add a string metadata column.
    pub fn add_string(mut self, name: &str, indexed: bool) -> Self {
        self.fields.push(MetadataField {
            name: name.to_string(),
            field_type: MetadataFieldType::String,
            indexed,
        });
        self
    }

    /// Add a numeric (f64) metadata column.
    pub fn add_number(mut self, name: &str, indexed: bool) -> Self {
        self.fields.push(MetadataField {
            name: name.to_string(),
            field_type: MetadataFieldType::Number,
            indexed,
        });
        self
    }

    /// Add a boolean metadata column.
    pub fn add_bool(mut self, name: &str, indexed: bool) -> Self {
        self.fields.push(MetadataField {
            name: name.to_string(),
            field_type: MetadataFieldType::Bool,
            indexed,
        });
        self
    }
}

/// Build the full Arrow schema for the LanceDB `chunks` table.
///
/// Fixed columns:
///   - `doc_uuid`: Utf8 (document UUID)
///   - `sub_idx`: UInt32 (sub-chunk index within document)
///   - `date`: Utf8 (ISO 8601)
///   - `body`: LargeUtf8 (full parent document body)
///   - `sub_body`: Utf8 (this sub-chunk's text, FTS indexed)
///   - `lens`: Utf8 (comma-separated codepoint lengths)
///   - `hash`: Utf8 (content hash for dedup)
///   - `vectors`: List(FixedSizeList(Float32, EMBEDDING_DIM)) (multivector column)
///
/// Plus user-defined metadata columns from the schema.
pub fn build_arrow_schema(ms: &MetadataSchema) -> Arc<Schema> {
    let dim = EMBEDDING_DIM as i32;

    let mut fields = vec![
        Field::new("doc_uuid", DataType::Utf8, false),
        Field::new("sub_idx", DataType::UInt32, false),
        Field::new("date", DataType::Utf8, false),
        Field::new("body", DataType::LargeUtf8, true),
        Field::new("sub_body", DataType::Utf8, false),
        Field::new("lens", DataType::Utf8, false),
        Field::new("hash", DataType::Utf8, false),
        // Multivector column: List of FixedSizeList<f32, 128>
        // Each row contains a variable number of 128-dim token embedding vectors.
        Field::new(
            "vectors",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(
                    Arc::new(Field::new("elem", DataType::Float32, true)),
                    dim,
                ),
                true,
            ))),
            false,
        ),
    ];

    for f in &ms.fields {
        let dt = match f.field_type {
            MetadataFieldType::String => DataType::Utf8,
            MetadataFieldType::Number => DataType::Float64,
            MetadataFieldType::Bool => DataType::Boolean,
        };
        fields.push(Field::new(&f.name, dt, true));
    }

    Arc::new(Schema::new(fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_schema() {
        let ms = MetadataSchema::new();
        let schema = build_arrow_schema(&ms);
        // 8 fixed columns
        assert_eq!(schema.fields().len(), 8);
        assert!(schema.field_with_name("doc_uuid").is_ok());
        assert!(schema.field_with_name("vectors").is_ok());
    }

    #[test]
    fn test_schema_with_metadata() {
        let ms = MetadataSchema::new()
            .add_string("session_id", true)
            .add_number("turn", false)
            .add_bool("is_active", false);
        let schema = build_arrow_schema(&ms);
        assert_eq!(schema.fields().len(), 11); // 8 fixed + 3 metadata
        assert_eq!(
            schema.field_with_name("session_id").unwrap().data_type(),
            &DataType::Utf8
        );
        assert_eq!(
            schema.field_with_name("turn").unwrap().data_type(),
            &DataType::Float64
        );
        assert_eq!(
            schema.field_with_name("is_active").unwrap().data_type(),
            &DataType::Boolean
        );
    }
}
