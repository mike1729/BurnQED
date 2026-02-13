//! EBM model components: energy head MLP, spectral normalization, and
//! tensor bridge between LLM encoder output and burn tensors.

pub mod bridge;
pub mod encoder;
pub mod energy_head;
pub mod spectral_norm;
