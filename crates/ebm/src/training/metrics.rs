/// EBM-specific training metrics with health checks.
#[derive(Debug, Clone)]
pub struct EBMMetrics {
    pub loss: f64,
    pub energy_gap: f64,
}
