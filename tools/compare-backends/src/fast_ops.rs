//! Simple replacement for fast_ops - just use standard tensor addition

use candle_core::{Result, Tensor};

pub fn fast_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    (a + b)
}
