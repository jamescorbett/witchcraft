// T5 Encoder implementation using ONNX Runtime
#![allow(dead_code)]

use crate::{embed_raw_asset, embed_zst_asset};
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use serde::Deserialize;
use std::cell::RefCell;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// Embed assets
embed_zst_asset!(pub CONFIG,    "config.json.zst");
embed_zst_asset!(pub TOKENIZER, "tokenizer.json.zst");
embed_raw_asset!(pub MODEL,     "xtr-encoder-128-int8.onnx");
embed_raw_asset!(pub WEIGHTS,   "xtr-f32.safetensors");

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}

/// Builder for T5 encoder model
pub struct T5ModelBuilder {
    config: Config,
}

impl T5ModelBuilder {
    pub fn load(assets: &PathBuf) -> Result<(Self, Tokenizer)> {
        // Load config
        let cfg_bytes = CONFIG.bytes(assets).map_err(|_| anyhow!("Failed to load config"))?;
        let config: Config = serde_json::from_slice(cfg_bytes)?;

        // Load tokenizer
        let tok_bytes = TOKENIZER.bytes(assets).map_err(|_| anyhow!("Failed to load tokenizer"))?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes).map_err(anyhow::Error::msg)?;

        Ok((Self { config }, tokenizer))
    }

    pub fn build_encoder(&self, _device: &Device, assets: &PathBuf) -> Result<T5EncoderModel> {
        T5EncoderModel::new(assets)
    }
}

/// T5 Encoder using ONNX Runtime backend
pub struct T5EncoderModel {
    session: RefCell<Session>,
    projection: Tensor,  // 768 → 128 linear projection
    device: Device,
}

impl T5EncoderModel {
    pub fn new(assets: &PathBuf) -> Result<Self> {
        log::info!("[ONNX] Creating T5 encoder...");

        // Load ONNX model
        let session = {
            #[cfg(feature = "embed-assets")]
            {
                let model_bytes = MODEL.bytes();
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(4)?
                    .commit_from_memory(model_bytes)?
            }

            #[cfg(not(feature = "embed-assets"))]
            {
                let model_path = MODEL.path(assets);
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(4)?
                    .commit_from_file(&model_path)?
            }
        };

        // Load projection layer (768 → 128) from safetensors using mmap
        let projection = {
            #[cfg(feature = "embed-assets")]
            {
                let weights_bytes = WEIGHTS.bytes();
                let temp = tempfile::NamedTempFile::new()?;
                std::fs::write(temp.path(), weights_bytes)?;

                let file = std::fs::File::open(temp.path())?;
                let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
                let projection_data = tensors.tensor("linear.weight")?;
                let projection_shape = projection_data.shape();
                let projection_f32: Vec<f32> = projection_data.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_slice(&projection_f32, (projection_shape[0], projection_shape[1]), &Device::Cpu)?
            }
            #[cfg(not(feature = "embed-assets"))]
            {
                let weights_path = WEIGHTS.path(assets);
                let file = std::fs::File::open(&weights_path)?;
                let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
                let projection_data = tensors.tensor("linear.weight")?;
                let projection_shape = projection_data.shape();
                let projection_f32: Vec<f32> = projection_data.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_slice(&projection_f32, (projection_shape[0], projection_shape[1]), &Device::Cpu)?
            }
        };

        log::info!("[ONNX] T5 encoder created with ONNX Runtime");

        Ok(Self {
            session: RefCell::new(session),
            projection,
            device: Device::Cpu,
        })
    }

    /// Encode input token IDs to embeddings
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Extract token IDs from tensor (tokenizer produces U32, convert to I64 for ONNX)
        let input_u32 = input.squeeze(0)?.to_vec1::<u32>()?;
        let input_ids: Vec<i64> = input_u32.iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();
        log::debug!("[ONNX] Encoding {} tokens", seq_len);

        // Create input tensor for ONNX Runtime
        let input_tensor = Value::from_array(([1, seq_len], input_ids))?;

        // Run inference and extract data immediately
        let inputs = ort::inputs![input_tensor];
        let (output_shape, output_data) = {
            let mut session = self.session.borrow_mut();
            let outputs = session.run(inputs)?;

            // Extract output tensor as ndarray
            let array_view = outputs[0].try_extract_array::<f32>()?;
            let shape = array_view.shape().to_vec();
            let data: Vec<f32> = array_view.iter().copied().collect();
            (shape, data)
        };

        // Convert to candle Tensor
        let encoder_output = Tensor::from_slice(&output_data, (output_shape[0], output_shape[1], output_shape[2]), &self.device)?;

        // Apply final projection (768 → 128)
        // Shape: [batch, seq_len, 768] @ [128, 768].T = [batch, seq_len, 128]
        let projected = encoder_output.broadcast_matmul(&self.projection.t()?)?;

        Ok(projected)
    }

    pub fn from_safetensors(_path: &std::path::Path, _device: &Device) -> Result<Self> {
        Err(anyhow!("Use T5EncoderModel::new with assets path instead"))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
