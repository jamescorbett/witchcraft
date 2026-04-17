use log::{debug, info, warn};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// Conditionally compile T5 encoder based on features
#[cfg(feature = "t5-quantized")]
pub mod quantized_t5;
#[cfg(feature = "t5-quantized")]
use quantized_t5 as t5_encoder;
pub mod fast_ops;
#[cfg(feature = "hybrid-dequant")]
pub mod fused_matmul;

#[cfg(feature = "t5-openvino")]
mod openvino_t5;
#[cfg(feature = "t5-openvino")]
use openvino_t5 as t5_encoder;

// Compile-time checks for mutual exclusivity
#[cfg(not(any(feature = "t5-quantized", feature = "t5-openvino")))]
compile_error!("Must enable exactly one T5 backend: t5-quantized or t5-openvino");

#[cfg(all(feature = "t5-quantized", feature = "t5-openvino"))]
compile_error!("Cannot enable multiple T5 backends simultaneously");

mod embedder;
pub use embedder::Embedder;

pub mod assets;

pub mod filter;
pub mod schema;
pub mod types;

pub use filter::{Filter, FilterCondition, FilterOp, MetadataValue};
pub use schema::{MetadataField, MetadataFieldType, MetadataSchema};
pub use types::{DocPtr, SearchResult};

#[allow(dead_code)]
mod priority;

#[allow(dead_code)]
mod progress_reporter;

#[allow(dead_code)]
mod histogram;

#[cfg(feature = "napi")]
#[allow(dead_code)]
mod napi;

use anyhow::Result;
use arrow_array::builder::{
    Float32Builder, Float64Builder, LargeStringBuilder, ListBuilder, StringBuilder, UInt32Builder,
};
use arrow_array::builder::{BooleanBuilder, FixedSizeListBuilder};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use candle_core::{Device, Tensor};
use futures::TryStreamExt;
use lancedb::index::scalar::{BTreeIndexBuilder, FtsIndexBuilder, FullTextSearchQuery};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::DistanceType;
use lru::LruCache;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const EMBEDDING_DIM: usize = 128;
const HASH_CHARS: usize = 32;
const TABLE_NAME: &str = "chunks";

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub fn make_device() -> Device {
    Device::new_metal(0).unwrap_or(Device::Cpu)
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
pub fn make_device() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// Embeddings Cache
// ---------------------------------------------------------------------------

pub struct EmbeddingsCache {
    cache: LruCache<String, Tensor>,
}

impl EmbeddingsCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(std::num::NonZeroUsize::new(capacity.max(1)).unwrap()),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<Tensor> {
        self.cache.get(key).cloned()
    }

    pub fn put(&mut self, key: &str, value: &Tensor) {
        self.cache.put(key.into(), value.clone());
    }
}

// ---------------------------------------------------------------------------
// Content hashing (preserved from old db.rs)
// ---------------------------------------------------------------------------

fn content_hash(body: &str, lens: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    hasher.update(lens.as_bytes());
    let result = hasher.finalize();
    let hex = format!("{:x}", result);
    hex[..HASH_CHARS].to_string()
}

// ---------------------------------------------------------------------------
// String splitting by codepoints (preserved)
// ---------------------------------------------------------------------------

pub fn split_by_codepoints<'a>(s: &'a str, lengths: &[usize]) -> Vec<&'a str> {
    let mut boundaries: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    boundaries.push(s.len());

    let char_len = boundaries.len() - 1;
    let sum_chars: usize = lengths.iter().copied().sum();
    if sum_chars != char_len {
        warn!("sum of lengths does not match utf8-length of string!");
        return vec![];
    }

    let mut parts = Vec::with_capacity(lengths.len());
    let mut pos = 0usize;

    for &chunk_chars in lengths {
        let start_byte = boundaries[pos];
        let end_pos = pos + chunk_chars;
        let end_byte = boundaries[end_pos];
        parts.push(&s[start_byte..end_byte]);
        pos = end_pos;
    }
    parts
}

// ---------------------------------------------------------------------------
// Reciprocal Rank Fusion (preserved, updated for new DocPtr type)
// ---------------------------------------------------------------------------

pub fn reciprocal_rank_fusion(list1: &[DocPtr], list2: &[DocPtr], k: f64) -> Vec<DocPtr> {
    let mut scores: HashMap<DocPtr, f64> = HashMap::new();

    for (rank, doc_id) in list1.iter().enumerate() {
        let score = 1.0 / (3.0 + k + rank as f64);
        *scores.entry(doc_id.clone()).or_insert(0.0) += score;
    }

    for (rank, doc_id) in list2.iter().enumerate() {
        let score = 1.0 / (k + rank as f64);
        *scores.entry(doc_id.clone()).or_insert(0.0) += score;
    }

    let mut results: Vec<(DocPtr, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.into_iter().map(|(idx, _)| idx).collect()
}

// ---------------------------------------------------------------------------
// Embedding splitting: assign token embeddings to sub-chunks
// ---------------------------------------------------------------------------

/// Given token-level embeddings and their byte offsets, plus the sub-chunk
/// codepoint lengths, split the token embeddings into groups per sub-chunk.
///
/// Returns a Vec of Vec<Vec<f32>> — one entry per sub-chunk, each containing
/// the token embeddings (as flat f32 vecs) belonging to that sub-chunk.
fn split_embeddings_by_chunks(
    embeddings: &Tensor,
    offsets: &[(usize, usize)],
    body: &str,
    lengths: &[usize],
) -> Result<Vec<Vec<Vec<f32>>>> {
    // Convert codepoint lengths to cumulative byte boundaries
    let mut cum_lengths: Vec<usize> = lengths.to_vec();
    for i in 1..cum_lengths.len() {
        cum_lengths[i] += cum_lengths[i - 1];
    }

    let embeddings = embeddings.squeeze(0)?; // [N, 128]
    let (num_tokens, _dim) = embeddings.dims2()?;

    let mut result: Vec<Vec<Vec<f32>>> = vec![vec![]; lengths.len()];
    let mut i = 0; // token index
    let mut j = 0; // chunk index

    // Convert codepoint cumulative lengths to byte boundaries
    let mut char_boundaries: Vec<usize> = body.char_indices().map(|(idx, _)| idx).collect();
    char_boundaries.push(body.len());

    let byte_boundaries: Vec<usize> = cum_lengths
        .iter()
        .map(|&cp| {
            if cp <= char_boundaries.len() - 1 {
                char_boundaries[cp]
            } else {
                body.len()
            }
        })
        .collect();

    while i < num_tokens && j < byte_boundaries.len() {
        let token_end_byte = offsets[i].1;
        let chunk_end_byte = byte_boundaries[j];

        if token_end_byte <= chunk_end_byte {
            let row = embeddings.get(i)?.to_vec1::<f32>()?;
            result[j].push(row);
            i += 1;
        } else {
            j += 1;
        }
    }
    // Any remaining tokens go into the last chunk
    while i < num_tokens {
        let last = result.len() - 1;
        let row = embeddings.get(i)?.to_vec1::<f32>()?;
        result[last].push(row);
        i += 1;
    }

    // Ensure every chunk has at least one embedding (use zero vector if needed)
    for chunk_vecs in &mut result {
        if chunk_vecs.is_empty() {
            chunk_vecs.push(vec![0.0f32; EMBEDDING_DIM]);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Arrow RecordBatch construction
// ---------------------------------------------------------------------------

fn build_record_batch(
    schema: &Arc<Schema>,
    metadata_schema: &MetadataSchema,
    rows: &[SubChunkRow],
) -> Result<RecordBatch> {
    let mut doc_uuid_builder = StringBuilder::new();
    let mut sub_idx_builder = UInt32Builder::new();
    let mut date_builder = StringBuilder::new();
    let mut body_builder = LargeStringBuilder::new();
    let mut sub_body_builder = StringBuilder::new();
    let mut lens_builder = StringBuilder::new();
    let mut hash_builder = StringBuilder::new();

    // Multivector builder: List<FixedSizeList<f32, 128>>
    let inner_builder = FixedSizeListBuilder::new(Float32Builder::new(), EMBEDDING_DIM as i32)
        .with_field(Field::new("elem", DataType::Float32, true));
    let mut vectors_builder = ListBuilder::new(inner_builder);

    // Metadata builders (one per field, in schema order)
    let mut meta_string_builders: Vec<(String, StringBuilder)> = vec![];
    let mut meta_number_builders: Vec<(String, Float64Builder)> = vec![];
    let mut meta_bool_builders: Vec<(String, BooleanBuilder)> = vec![];

    for f in &metadata_schema.fields {
        match f.field_type {
            MetadataFieldType::String => {
                meta_string_builders.push((f.name.clone(), StringBuilder::new()));
            }
            MetadataFieldType::Number => {
                meta_number_builders.push((f.name.clone(), Float64Builder::new()));
            }
            MetadataFieldType::Bool => {
                meta_bool_builders.push((f.name.clone(), BooleanBuilder::new()));
            }
        }
    }

    for row in rows {
        doc_uuid_builder.append_value(&row.doc_uuid);
        sub_idx_builder.append_value(row.sub_idx);
        date_builder.append_value(&row.date);
        body_builder.append_value(&row.body);
        sub_body_builder.append_value(&row.sub_body);
        lens_builder.append_value(&row.lens);
        hash_builder.append_value(&row.hash);

        // Build multivector: each token embedding is a FixedSizeList<f32, 128>
        let fsl_builder = vectors_builder.values();
        for token_vec in &row.vectors {
            let vals_builder = fsl_builder.values();
            for &v in token_vec {
                vals_builder.append_value(v);
            }
            fsl_builder.append(true);
        }
        vectors_builder.append(true);

        // Metadata columns
        for (name, builder) in &mut meta_string_builders {
            match row.metadata.get(name.as_str()) {
                Some(MetadataValue::String(s)) => builder.append_value(s),
                _ => builder.append_null(),
            }
        }
        for (name, builder) in &mut meta_number_builders {
            match row.metadata.get(name.as_str()) {
                Some(MetadataValue::Number(n)) => builder.append_value(*n),
                _ => builder.append_null(),
            }
        }
        for (name, builder) in &mut meta_bool_builders {
            match row.metadata.get(name.as_str()) {
                Some(MetadataValue::Bool(b)) => builder.append_value(*b),
                _ => builder.append_null(),
            }
        }
    }

    // Assemble columns in schema order
    let mut columns: Vec<Arc<dyn arrow_array::Array>> = vec![
        Arc::new(doc_uuid_builder.finish()),
        Arc::new(sub_idx_builder.finish()),
        Arc::new(date_builder.finish()),
        Arc::new(body_builder.finish()),
        Arc::new(sub_body_builder.finish()),
        Arc::new(lens_builder.finish()),
        Arc::new(hash_builder.finish()),
        Arc::new(vectors_builder.finish()),
    ];

    // Add metadata columns in the same order as the schema
    for f in &metadata_schema.fields {
        match f.field_type {
            MetadataFieldType::String => {
                let (_, builder) = meta_string_builders
                    .iter_mut()
                    .find(|(n, _)| n == &f.name)
                    .unwrap();
                columns.push(Arc::new(builder.finish()));
            }
            MetadataFieldType::Number => {
                let (_, builder) = meta_number_builders
                    .iter_mut()
                    .find(|(n, _)| n == &f.name)
                    .unwrap();
                columns.push(Arc::new(builder.finish()));
            }
            MetadataFieldType::Bool => {
                let (_, builder) = meta_bool_builders
                    .iter_mut()
                    .find(|(n, _)| n == &f.name)
                    .unwrap();
                columns.push(Arc::new(builder.finish()));
            }
        }
    }

    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

/// Internal representation of a sub-chunk row before Arrow conversion.
struct SubChunkRow {
    doc_uuid: String,
    sub_idx: u32,
    date: String,
    body: String,
    sub_body: String,
    lens: String,
    hash: String,
    vectors: Vec<Vec<f32>>, // token embeddings for this sub-chunk
    metadata: HashMap<String, MetadataValue>,
}

// ---------------------------------------------------------------------------
// Witchcraft: the main API struct
// ---------------------------------------------------------------------------

pub struct Witchcraft {
    #[allow(dead_code)]
    db: lancedb::Connection,
    table: lancedb::Table,
    embedder: Embedder,
    cache: EmbeddingsCache,
    metadata_schema: MetadataSchema,
    arrow_schema: Arc<Schema>,
}

impl Witchcraft {
    /// Open or create a Witchcraft database at the given path.
    ///
    /// If the table does not exist, it is created with the given schema.
    /// The `assets` path should point to the T5 model weights directory.
    pub async fn new(
        path: impl AsRef<Path>,
        assets: impl AsRef<Path>,
        metadata_schema: MetadataSchema,
    ) -> Result<Self> {
        let device = make_device();
        let embedder = Embedder::new(&device, assets.as_ref())?;
        let db = lancedb::connect(path.as_ref().to_str().unwrap())
            .execute()
            .await?;

        let arrow_schema = schema::build_arrow_schema(&metadata_schema);
        let table = match db.open_table(TABLE_NAME).execute().await {
            Ok(t) => t,
            Err(_) => {
                db.create_empty_table(TABLE_NAME, arrow_schema.clone())
                    .execute()
                    .await?
            }
        };

        Ok(Self {
            db,
            table,
            embedder,
            cache: EmbeddingsCache::new(32),
            metadata_schema,
            arrow_schema,
        })
    }

    /// Get a reference to the underlying LanceDB table for direct queries.
    pub fn table_ref(&self) -> &lancedb::Table {
        &self.table
    }

    /// Open read-only (no embedder needed if only searching with cached index).
    pub async fn open(
        path: impl AsRef<Path>,
        assets: impl AsRef<Path>,
        metadata_schema: MetadataSchema,
    ) -> Result<Self> {
        // For now, always create an embedder (needed for query embedding).
        Self::new(path, assets, metadata_schema).await
    }

    // -----------------------------------------------------------------------
    // Document operations
    // -----------------------------------------------------------------------

    /// Add a document, embedding it immediately and writing sub-chunk rows.
    pub async fn add_document(
        &mut self,
        uuid: &Uuid,
        date: Option<&str>,
        metadata: HashMap<String, MetadataValue>,
        body: &str,
        lens: Option<Vec<usize>>,
    ) -> Result<()> {
        let date_str = date.unwrap_or("").to_string();

        let lengths = match lens {
            Some(l) => l,
            None => vec![body.chars().count()],
        };
        let lens_str: String = lengths
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let hash = content_hash(body, &lens_str);

        // Content-addressable dedup: skip if hash already exists
        let existing = self
            .table
            .query()
            .only_if(format!("hash = '{}'", hash))
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        if !existing.is_empty() {
            debug!("skipping duplicate content hash {}", hash);
            return Ok(());
        }

        // Embed the full body
        let (embeddings_tensor, offsets) = self.embedder.embed(body)?;

        // Split embeddings into per-sub-chunk groups
        let chunk_embeddings =
            split_embeddings_by_chunks(&embeddings_tensor, &offsets, body, &lengths)?;

        // Split body text into sub-chunks
        let sub_bodies = split_by_codepoints(body, &lengths);

        // Build rows
        let mut rows = Vec::with_capacity(lengths.len());
        for (i, (sub_body, vectors)) in sub_bodies.iter().zip(chunk_embeddings).enumerate() {
            rows.push(SubChunkRow {
                doc_uuid: uuid.to_string(),
                sub_idx: i as u32,
                date: date_str.clone(),
                body: body.to_string(),
                sub_body: sub_body.to_string(),
                lens: lens_str.clone(),
                hash: hash.clone(),
                vectors,
                metadata: metadata.clone(),
            });
        }

        let batch = build_record_batch(&self.arrow_schema, &self.metadata_schema, &rows)?;
        self.table.add(batch).execute().await?;

        debug!(
            "added document {} with {} sub-chunks",
            uuid,
            rows.len()
        );
        Ok(())
    }

    /// Remove all rows for a given document UUID.
    pub async fn remove_document(&mut self, uuid: &Uuid) -> Result<()> {
        self.table
            .delete(&format!("doc_uuid = '{}'", uuid))
            .await?;
        Ok(())
    }

    /// Delete documents matching a filter.
    pub async fn delete_with_filter(&mut self, filter: &Filter) -> Result<()> {
        let filter_str = filter.to_lance_filter();
        if filter_str.is_empty() {
            self.clear().await?;
        } else {
            self.table.delete(&filter_str).await?;
        }
        Ok(())
    }

    /// Clear all data.
    pub async fn clear(&mut self) -> Result<()> {
        self.table.delete("true").await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Index management
    // -----------------------------------------------------------------------

    /// Build vector + FTS + scalar indexes.
    ///
    /// Call after bulk ingestion. LanceDB handles the IVF construction
    /// internally -- this replaces the entire kmeans/bucket pipeline.
    pub async fn build_index(&mut self) -> Result<()> {
        info!("building vector index (IVF_PQ with cosine)...");
        // Vector index on multivector column
        self.table
            .create_index(
                &["vectors"],
                lancedb::index::Index::Auto,
            )
            .execute()
            .await?;

        info!("building FTS index on sub_body...");
        self.table
            .create_index(
                &["sub_body"],
                lancedb::index::Index::FTS(FtsIndexBuilder::default()),
            )
            .execute()
            .await?;

        // Scalar indexes on indexed metadata columns
        for field in &self.metadata_schema.fields {
            if field.indexed {
                info!("building scalar index on {}...", field.name);
                self.table
                    .create_index(
                        &[&field.name],
                        lancedb::index::Index::BTree(BTreeIndexBuilder::default()),
                    )
                    .execute()
                    .await?;
            }
        }

        info!("all indexes built successfully");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Hybrid search combining multivector semantic search and FTS with RRF fusion.
    pub async fn search(
        &mut self,
        query: &str,
        _threshold: f32,
        top_k: usize,
        use_fulltext: bool,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let now = std::time::Instant::now();
        let q = query.split_whitespace().collect::<Vec<_>>().join(" ");

        let filter_str = filter.map(|f| f.to_lance_filter()).unwrap_or_default();

        // -- Semantic search via LanceDB multivector --
        let sem_matches = if q.len() > 3 {
            let qe = match self.cache.get(&q) {
                Some(existing) => existing,
                None => {
                    let (qe, _) = self.embedder.embed(&q)?;
                    let qe = qe.get(0)?;
                    self.cache.put(&q, &qe);
                    qe
                }
            };

            self.semantic_search(&qe, top_k * 2, &filter_str).await?
        } else {
            vec![]
        };

        // -- Full-text search --
        let fts_matches = if use_fulltext {
            self.fulltext_search(&q, top_k * 2, &filter_str).await?
        } else {
            vec![]
        };

        // -- Fusion --
        let sem_ptrs: Vec<DocPtr> = sem_matches
            .iter()
            .map(|(ptr, _)| ptr.clone())
            .collect();
        info!("semantic search found {} matches", sem_ptrs.len());

        let mut all_scores: HashMap<DocPtr, f32> = HashMap::new();
        for (ptr, score) in &sem_matches {
            all_scores.insert(ptr.clone(), *score);
        }
        for (ptr, score) in &fts_matches {
            all_scores.insert(ptr.clone(), *score);
        }

        let mut fused = if use_fulltext {
            let fts_ptrs: Vec<DocPtr> = fts_matches
                .iter()
                .map(|(ptr, _)| ptr.clone())
                .collect();
            reciprocal_rank_fusion(&fts_ptrs, &sem_ptrs, 60.0)
        } else {
            sem_ptrs
        };
        fused.truncate(top_k);

        // -- Assemble results --
        // We need to fetch full document data for each result.
        // The data is already in the table rows, so we query for each hit.
        let mut results = Vec::with_capacity(fused.len());
        for (doc_uuid, sub_idx) in &fused {
            let score = all_scores
                .get(&(doc_uuid.clone(), *sub_idx))
                .copied()
                .unwrap_or(0.0);

            // Fetch the full row data
            let row_filter = format!("doc_uuid = '{}' AND sub_idx = {}", doc_uuid, sub_idx);
            let batches = self
                .table
                .query()
                .only_if(&row_filter)
                .select(Select::Columns(vec![
                    "body".to_string(),
                    "lens".to_string(),
                    "date".to_string(),
                ]))
                .limit(1)
                .execute()
                .await?
                .try_collect::<Vec<_>>()
                .await?;

            if let Some(batch) = batches.first() {
                if batch.num_rows() > 0 {
                    let body_col = batch
                        .column_by_name("body")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::LargeStringArray>());
                    let lens_col = batch
                        .column_by_name("lens")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
                    let date_col = batch
                        .column_by_name("date")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());

                    if let (Some(body_arr), Some(lens_arr), Some(date_arr)) =
                        (body_col, lens_col, date_col)
                    {
                        let body = body_arr.value(0);
                        let lens_str = lens_arr.value(0);
                        let date = date_arr.value(0).to_string();

                        let lens: Vec<usize> = lens_str
                            .split(',')
                            .filter_map(|s| s.parse::<usize>().ok())
                            .collect();
                        let bodies: Vec<String> = split_by_codepoints(body, &lens)
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect();

                        let sub = (*sub_idx as usize).min(bodies.len().saturating_sub(1)) as u32;

                        let meta = HashMap::new();

                        results.push(SearchResult {
                            score,
                            metadata: meta,
                            bodies,
                            matched_sub_idx: sub,
                            date,
                        });
                    }
                }
            }
        }

        // Ensure monotonically non-increasing scores
        let mut max = -1.0f32;
        for r in results.iter_mut().rev() {
            max = max.max(r.score);
            r.score = max;
        }

        debug!(
            "witchcraft search took {} ms end-to-end.",
            now.elapsed().as_millis()
        );
        Ok(results)
    }

    /// Multivector semantic search via LanceDB.
    ///
    /// Each query token embedding is passed as a separate query vector using
    /// `nearest_to()` + `add_query_vector()`. LanceDB dispatches this as a
    /// multivector (MaxSim/late-interaction) search against the `vectors` column
    /// which stores `List<FixedSizeList<f32, 128>>` per sub-chunk.
    async fn semantic_search(
        &self,
        query_embeddings: &Tensor,
        limit: usize,
        filter_str: &str,
    ) -> Result<Vec<(DocPtr, f32)>> {
        let (num_tokens, _dim) = query_embeddings.dims2()?;

        // Convert query tensor to per-token Vec<f32> for LanceDB multivector query
        let mut query_vecs: Vec<Vec<f32>> = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            let row = query_embeddings.get(i)?.to_vec1::<f32>()?;
            query_vecs.push(row);
        }

        // Build multivector query: first token via nearest_to, rest via add_query_vector
        // This triggers LanceDB's multivector (MaxSim) scoring against the
        // List<FixedSizeList<f32, 128>> column.
        let first = query_vecs[0].clone();
        let mut qb = self
            .table
            .query()
            .nearest_to(first)?
            .column("vectors")
            .distance_type(DistanceType::Cosine)
            .limit(limit);

        for token_vec in &query_vecs[1..] {
            qb = qb.add_query_vector(token_vec.clone())?;
        }

        if !filter_str.is_empty() {
            qb = qb.only_if(filter_str);
        }

        let batches = qb
            .select(Select::Columns(vec![
                "doc_uuid".to_string(),
                "sub_idx".to_string(),
                "_distance".to_string(),
            ]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut results = Vec::new();
        for batch in &batches {
            let uuid_col = batch
                .column_by_name("doc_uuid")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            let idx_col = batch
                .column_by_name("sub_idx")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>());
            let dist_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());

            if let (Some(uuids), Some(idxs), Some(dists)) = (uuid_col, idx_col, dist_col) {
                for i in 0..batch.num_rows() {
                    let uuid = uuids.value(i).to_string();
                    let sub_idx = idxs.value(i);
                    let distance = dists.value(i);
                    // Convert cosine distance to similarity score (1 - distance)
                    let score = 1.0 - distance;
                    results.push(((uuid, sub_idx), score));
                }
            }
        }

        Ok(results)
    }

    /// Full-text search via LanceDB FTS index.
    async fn fulltext_search(
        &self,
        query: &str,
        limit: usize,
        filter_str: &str,
    ) -> Result<Vec<(DocPtr, f32)>> {
        // Sanitize query to alphanumeric + whitespace
        let sanitized: String = query
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        if sanitized.trim().is_empty() {
            return Ok(vec![]);
        }

        let fts_query = FullTextSearchQuery::new(sanitized.clone());
        let mut qb = self.table.query().full_text_search(fts_query).limit(limit);

        if !filter_str.is_empty() {
            qb = qb.only_if(filter_str);
        }

        let batches = qb
            .select(Select::Columns(vec![
                "doc_uuid".to_string(),
                "sub_idx".to_string(),
                "sub_body".to_string(),
            ]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut results = Vec::new();
        for batch in &batches {
            let uuid_col = batch
                .column_by_name("doc_uuid")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            let idx_col = batch
                .column_by_name("sub_idx")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>());
            let body_col = batch
                .column_by_name("sub_body")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());

            if let (Some(uuids), Some(idxs), Some(bodies)) = (uuid_col, idx_col, body_col) {
                for i in 0..batch.num_rows() {
                    let uuid = uuids.value(i).to_string();
                    let sub_idx = idxs.value(i);
                    let sub_body = bodies.value(i);
                    // Score with Jaro-Winkler similarity (same as original)
                    let score = strsim::jaro_winkler(&sanitized, sub_body) as f32;
                    results.push(((uuid, sub_idx), score));
                }
            }
        }

        info!("full text found {} matches", results.len());
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Scoring (no DB needed)
    // -----------------------------------------------------------------------

    /// Score a list of sentences against a query using the embedding model.
    ///
    /// Returns a score per sentence (higher = more relevant).
    pub fn score_sentences(
        &mut self,
        query: &str,
        sentences: &[String],
    ) -> Result<Vec<f32>> {
        score_query_sentences(&self.embedder, &mut self.cache, &query.to_string(), sentences)
    }
}

// ---------------------------------------------------------------------------
// Standalone scoring function (no DB needed)
// ---------------------------------------------------------------------------

fn split_tensor(tensor: &Tensor) -> Vec<Tensor> {
    let num_rows = tensor.dim(0).unwrap();
    (0..num_rows)
        .map(|i| {
            let row_tensor = tensor.get(i).unwrap();
            row_tensor.unsqueeze(0).unwrap()
        })
        .collect()
}

pub fn score_query_sentences(
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    q: &String,
    sentences: &[String],
) -> Result<Vec<f32>> {
    let now = std::time::Instant::now();
    let qe = match cache.get(q) {
        Some(existing) => existing,
        None => {
            let (qe, _offsets) = embedder.embed(q)?;
            qe.get(0)?
        }
    };
    let mut sizes = vec![];
    let mut ses = vec![];
    for s in sentences.iter() {
        let (se, _offsets) = embedder.embed(s)?;
        let se = se.get(0)?;
        let split = split_tensor(&se);
        sizes.push(split.len());
        ses.extend(split);
    }
    let ses = Tensor::cat(&ses, 0)?;
    let sim = fast_ops::matmul_t(&ses, &qe)?;
    let sim = sim.to_device(&Device::Cpu)?;

    let mut scores = vec![];
    let mut i = 0;
    for sz in sizes.iter() {
        let sz = *sz;
        let mut max = sim.get(i)?;
        for j in 1usize..sz {
            let row = sim.get(i + j)?;
            max = max.maximum(&row)?;
        }
        scores.push(max.mean(0)?.to_scalar::<f32>()?);
        i += sz;
    }
    debug!(
        "scoring {} sentences took {} ms.",
        sentences.len(),
        now.elapsed().as_millis()
    );
    Ok(scores)
}

#[cfg(test)]
mod tests;
