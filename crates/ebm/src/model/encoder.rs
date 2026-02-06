/// EncoderBackend: configurable encoder (shared 7B or dedicated 1.3B).
pub enum EncoderBackend {
    /// Share the policy model's backbone (7B, loaded in candle).
    Shared,
    /// Separate smaller encoder.
    Dedicated,
}
