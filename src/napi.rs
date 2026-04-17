//! Node.js NAPI bindings for Witchcraft.
//!
//! Provides an async API for Node.js via napi-rs.
//! All operations are async and return Promises to JavaScript.

#[cfg(feature = "napi")]
use napi::bindgen_prelude::*;
#[cfg(feature = "napi")]
use napi_derive::napi;

#[cfg(feature = "napi")]
use std::collections::HashMap;
#[cfg(feature = "napi")]
use std::sync::Arc;
#[cfg(feature = "napi")]
use tokio::sync::Mutex;

#[cfg(feature = "napi")]
use crate::{Filter, FilterOp, MetadataSchema, MetadataValue, Witchcraft as WitchcraftInner};

// ---------------------------------------------------------------------------
// Progress / logging callbacks (simplified stubs for now)
// ---------------------------------------------------------------------------

#[cfg(feature = "napi")]
pub fn progress_update(_progress: f64, _phase: &str) {
    // TODO: implement JS callback
}

#[cfg(feature = "napi")]
#[napi(object)]
pub struct SearchResultJs {
    pub score: f64,
    pub body: Vec<String>,
    pub idx: u32,
    pub date: String,
}

#[cfg(feature = "napi")]
#[napi]
pub struct WitchcraftNapi {
    inner: Arc<Mutex<WitchcraftInner>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "napi")]
#[napi]
impl WitchcraftNapi {
    #[napi(constructor)]
    pub fn new(db_path: String, assets: String, schema_json: String) -> Result<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| Error::from_reason(format!("failed to create tokio runtime: {}", e)))?,
        );

        let metadata_schema: MetadataSchema = serde_json::from_str(&schema_json)
            .map_err(|e| Error::from_reason(format!("invalid schema JSON: {}", e)))?;

        let inner = runtime.block_on(async {
            WitchcraftInner::new(db_path, assets, metadata_schema)
                .await
                .map_err(|e| Error::from_reason(format!("failed to initialize: {}", e)))
        })?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            runtime,
        })
    }

    #[napi]
    pub async fn add(
        &self,
        uuid: String,
        date: Option<String>,
        metadata_json: String,
        body: String,
        lengths: Option<Vec<u32>>,
    ) -> Result<()> {
        let inner = self.inner.clone();
        let uuid = uuid::Uuid::parse_str(&uuid)
            .map_err(|e| Error::from_reason(format!("invalid UUID: {}", e)))?;
        let metadata: HashMap<String, MetadataValue> = serde_json::from_str(&metadata_json)
            .unwrap_or_default();
        let lens = lengths.map(|l| l.into_iter().map(|v| v as usize).collect());

        let mut wc = inner.lock().await;
        wc.add_document(&uuid, date.as_deref(), metadata, &body, lens)
            .await
            .map_err(|e| Error::from_reason(format!("{}", e)))
    }

    #[napi]
    pub async fn search(
        &self,
        q: String,
        threshold: f64,
        top_k: u32,
        use_fulltext: bool,
    ) -> Result<Vec<SearchResultJs>> {
        let inner = self.inner.clone();
        let mut wc = inner.lock().await;
        let results = wc
            .search(&q, threshold as f32, top_k as usize, use_fulltext, None)
            .await
            .map_err(|e| Error::from_reason(format!("{}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| SearchResultJs {
                score: r.score as f64,
                body: r.bodies,
                idx: r.matched_sub_idx,
                date: r.date,
            })
            .collect())
    }

    #[napi]
    pub async fn remove(&self, uuid: String) -> Result<()> {
        let inner = self.inner.clone();
        let uuid = uuid::Uuid::parse_str(&uuid)
            .map_err(|e| Error::from_reason(format!("invalid UUID: {}", e)))?;
        let mut wc = inner.lock().await;
        wc.remove_document(&uuid)
            .await
            .map_err(|e| Error::from_reason(format!("{}", e)))
    }

    #[napi]
    pub async fn build_index(&self) -> Result<()> {
        let inner = self.inner.clone();
        let mut wc = inner.lock().await;
        wc.build_index()
            .await
            .map_err(|e| Error::from_reason(format!("{}", e)))
    }

    #[napi]
    pub async fn clear(&self) -> Result<()> {
        let inner = self.inner.clone();
        let mut wc = inner.lock().await;
        wc.clear()
            .await
            .map_err(|e| Error::from_reason(format!("{}", e)))
    }
}

// No-op when napi feature is not enabled
#[cfg(not(feature = "napi"))]
pub fn progress_update(_progress: f64, _phase: &str) {}
