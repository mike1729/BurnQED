use burn::module::Param;
use burn::prelude::*;

use crate::model::spectral_norm::{SpectralNormLinear, SpectralNormLinearConfig};

/// Configuration for the EnergyHead MLP.
///
/// Maps encoder hidden states (d_encoder) to scalar energy via a 4-layer
/// spectral-normed SiLU MLP with dropout and optional learnable temperature.
///
/// ```text
/// (batch, d_encoder)
///   → SpectralNormLinear(d_encoder→d_hidden1) → SiLU → Dropout
///   → SpectralNormLinear(d_hidden1→d_hidden2) → SiLU → Dropout
///   → SpectralNormLinear(d_hidden2→d_hidden3) → SiLU → Dropout
///   → SpectralNormLinear(d_hidden3→1, no bias) → squeeze
///   → raw_energy: (batch,)
/// ```
///
/// Temperature scaling is **not** applied in `forward()`. For InfoNCE loss
/// (which needs temperature to set softmax scale), call `temperature_scale()`
/// on the output. Margin ranking loss must use raw energies — temperature
/// would let the optimizer collapse τ→0 to trivially satisfy the margin.
#[derive(Config, Debug)]
pub struct EnergyHeadConfig {
    /// Encoder output dimension (e.g. 4096 for DeepSeek-Prover-V2-7B).
    pub d_encoder: usize,
    /// First hidden layer dimension.
    #[config(default = 2048)]
    pub d_hidden1: usize,
    /// Second hidden layer dimension.
    #[config(default = 1024)]
    pub d_hidden2: usize,
    /// Third hidden layer dimension.
    #[config(default = 512)]
    pub d_hidden3: usize,
    /// Dropout probability applied after each SiLU activation.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Apply spectral normalization to linear layers. When false, uses plain
    /// linear layers (more capacity, less regularization). Default: false.
    #[config(default = false)]
    pub spectral_norm: bool,
    /// Initial value for log_temperature. Default: -2.3 (temperature ≈ 0.1,
    /// effectively 10x energy amplification in InfoNCE).
    #[config(default = -2.3)]
    pub init_log_temperature: f64,
}

/// Energy head: spectral-normed MLP mapping encoder output to scalar energy.
///
/// The only trainable component in the EBM — the 7B encoder is frozen.
/// Lower energy = more provable state (by convention).
#[derive(Module, Debug)]
pub struct EnergyHead<B: Backend> {
    /// First spectral-normed linear: d_encoder → d_hidden1.
    sn_linear1: SpectralNormLinear<B>,
    /// Second spectral-normed linear: d_hidden1 → d_hidden2.
    sn_linear2: SpectralNormLinear<B>,
    /// Third spectral-normed linear: d_hidden2 → d_hidden3.
    sn_linear3: SpectralNormLinear<B>,
    /// Output spectral-normed linear: d_hidden3 → 1 (no bias).
    sn_linear4: SpectralNormLinear<B>,
    /// Dropout after first SiLU.
    dropout1: burn::nn::Dropout,
    /// Dropout after second SiLU.
    dropout2: burn::nn::Dropout,
    /// Dropout after third SiLU.
    dropout3: burn::nn::Dropout,
    /// Log-temperature parameter. Energy is divided by exp(log_temperature).
    /// Initialized to 0 so initial temperature = 1.
    log_temperature: Param<Tensor<B, 1>>,
}

impl EnergyHeadConfig {
    /// Initialize an EnergyHead with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> EnergyHead<B> {
        let sn = self.spectral_norm;
        EnergyHead {
            sn_linear1: SpectralNormLinearConfig::new(self.d_encoder, self.d_hidden1)
                .with_spectral_norm(sn)
                .init(device),
            sn_linear2: SpectralNormLinearConfig::new(self.d_hidden1, self.d_hidden2)
                .with_spectral_norm(sn)
                .init(device),
            sn_linear3: SpectralNormLinearConfig::new(self.d_hidden2, self.d_hidden3)
                .with_spectral_norm(sn)
                .init(device),
            sn_linear4: SpectralNormLinearConfig::new(self.d_hidden3, 1)
                .with_bias(false)
                .with_spectral_norm(sn)
                .init(device),
            dropout1: burn::nn::DropoutConfig::new(self.dropout).init(),
            dropout2: burn::nn::DropoutConfig::new(self.dropout).init(),
            dropout3: burn::nn::DropoutConfig::new(self.dropout).init(),
            log_temperature: Param::from_tensor(Tensor::from_floats(
                [self.init_log_temperature as f32],
                device,
            )),
        }
    }
}

impl<B: Backend> EnergyHead<B> {
    /// Forward pass: maps encoder hidden states to raw scalar energy values.
    ///
    /// Input shape: `(batch, d_encoder)`
    /// Output shape: `(batch,)`
    ///
    /// Returns **raw** (unscaled) energy. For InfoNCE loss, pipe through
    /// [`temperature_scale`](Self::temperature_scale) afterwards.
    pub fn forward(&self, h: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = self.sn_linear1.forward(h);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout1.forward(x);

        let x = self.sn_linear2.forward(x);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout2.forward(x);

        let x = self.sn_linear3.forward(x);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout3.forward(x);

        self.sn_linear4.forward(x).squeeze::<1>(1)
    }

    /// Apply learnable temperature scaling to energies.
    ///
    /// Returns `energy / exp(log_temperature)`. Only appropriate for
    /// InfoNCE loss (softmax has no inherent scale). **Never** use with
    /// margin ranking loss — the optimizer will collapse τ→0 to trivially
    /// satisfy the margin instead of learning real energy gaps.
    pub fn temperature_scale(&self, energy: Tensor<B, 1>) -> Tensor<B, 1> {
        let temperature = self.log_temperature.val().clamp(-5.0, 2.0).exp();
        energy / temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::tensor::Distribution;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_forward_shape() {
        let device = Default::default();
        let model = EnergyHeadConfig::new(4096).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::random(
            [8, 4096],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);
        assert_eq!(output.dims(), [8]);
    }

    #[test]
    fn test_small_model() {
        let device = Default::default();
        let model = EnergyHeadConfig::new(32)
            .with_d_hidden1(16)
            .with_d_hidden2(8)
            .with_d_hidden3(4)
            .init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::random(
            [4, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);
        assert_eq!(output.dims(), [4]);
    }

    #[test]
    fn test_different_inputs_different_energies() {
        let device = Default::default();
        let model = EnergyHeadConfig::new(32)
            .with_d_hidden1(16)
            .with_d_hidden2(8)
            .with_d_hidden3(4)
            .with_dropout(0.0)
            .init::<TestBackend>(&device);

        let input1 = Tensor::<TestBackend, 2>::random(
            [4, 32],
            Distribution::Normal(5.0, 1.0),
            &device,
        );
        let input2 = Tensor::<TestBackend, 2>::random(
            [4, 32],
            Distribution::Normal(-5.0, 1.0),
            &device,
        );

        let energy1 = model.forward(input1);
        let energy2 = model.forward(input2);

        let diff: f32 = (energy1 - energy2).abs().sum().into_scalar().elem();
        assert!(
            diff > 1e-6,
            "Different inputs should produce different energies, diff={diff}"
        );
    }

    #[test]
    fn test_gradient_flows_through_all_layers() {
        use burn::optim::GradientsParams;

        let device = Default::default();
        let model = EnergyHeadConfig::new(32)
            .with_d_hidden1(16)
            .with_d_hidden2(8)
            .with_d_hidden3(4)
            .init::<TestAutodiffBackend>(&device);

        let input = Tensor::<TestAutodiffBackend, 2>::random(
            [4, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);
        let loss = output.sum();

        let grads = GradientsParams::from_grads(loss.backward(), &model);

        // Check gradient flows to layer 1
        let grad1 = grads
            .get::<NdArray<f32>, 2>(model.sn_linear1.weight.id)
            .expect("sn_linear1 weight should have gradient");
        let grad1_sum: f32 = grad1.abs().sum().into_scalar().elem();
        assert!(
            grad1_sum > 0.0,
            "sn_linear1 gradient is zero — gradient not flowing"
        );

        // Check gradient flows to layer 2
        let grad2 = grads
            .get::<NdArray<f32>, 2>(model.sn_linear2.weight.id)
            .expect("sn_linear2 weight should have gradient");
        let grad2_sum: f32 = grad2.abs().sum().into_scalar().elem();
        assert!(
            grad2_sum > 0.0,
            "sn_linear2 gradient is zero — gradient not flowing"
        );

        // Check gradient flows to layer 3
        let grad3 = grads
            .get::<NdArray<f32>, 2>(model.sn_linear3.weight.id)
            .expect("sn_linear3 weight should have gradient");
        let grad3_sum: f32 = grad3.abs().sum().into_scalar().elem();
        assert!(
            grad3_sum > 0.0,
            "sn_linear3 gradient is zero — gradient not flowing"
        );

        // Check gradient flows to layer 4 (output)
        let grad4 = grads
            .get::<NdArray<f32>, 2>(model.sn_linear4.weight.id)
            .expect("sn_linear4 weight should have gradient");
        let grad4_sum: f32 = grad4.abs().sum().into_scalar().elem();
        assert!(
            grad4_sum > 0.0,
            "sn_linear4 gradient is zero — gradient not flowing"
        );

        // log_temperature only gets gradients via temperature_scale(), not forward().
        // Verify it gets gradient when temperature_scale is in the compute graph.
        let input2 = Tensor::<TestAutodiffBackend, 2>::random(
            [4, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let raw = model.forward(input2);
        let scaled = model.temperature_scale(raw);
        let loss2 = scaled.sum();
        let grads2 = GradientsParams::from_grads(loss2.backward(), &model);
        let temp_grad = grads2
            .get::<NdArray<f32>, 1>(model.log_temperature.id)
            .expect("log_temperature should have gradient via temperature_scale");
        let temp_grad_sum: f32 = temp_grad.abs().sum().into_scalar().elem();
        assert!(
            temp_grad_sum > 0.0,
            "log_temperature gradient is zero — gradient not flowing through temperature_scale"
        );
    }

    #[test]
    fn test_temperature_scaling() {
        // Temperature scaling test: verify that temperature_scale() applies
        // the learned temperature. forward() returns raw energy; temperature
        // is only used via temperature_scale() for InfoNCE.
        let device = Default::default();

        let model = EnergyHeadConfig::new(32)
            .with_d_hidden1(16)
            .with_d_hidden2(8)
            .with_d_hidden3(4)
            .with_dropout(0.0)
            .with_init_log_temperature(0.0) // temperature = 1.0
            .init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [8, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // forward() returns raw energy (no temperature)
        let raw_energy = model.forward(input.clone());
        // temperature_scale() divides by exp(0) = 1.0, so should be identical
        let scaled_energy = model.temperature_scale(raw_energy.clone());

        let diff: f32 = (raw_energy.clone() - scaled_energy).abs().sum().into_scalar().elem();
        assert!(
            diff < 1e-6,
            "With log_temperature=0 (temp=1), scaled should equal raw, diff={diff}"
        );

        // Build model with log_temperature = ln(2) → temperature = 2
        let model_temp2 = EnergyHeadConfig::new(32)
            .with_d_hidden1(16)
            .with_d_hidden2(8)
            .with_d_hidden3(4)
            .with_dropout(0.0)
            .with_init_log_temperature(2.0_f64.ln())
            .init::<TestBackend>(&device);

        // temperature_scale divides by 2 → half the magnitude
        let scaled = model_temp2.temperature_scale(raw_energy.clone());
        let expected = raw_energy / 2.0;
        let diff: f32 = (scaled - expected).abs().sum().into_scalar().elem();
        assert!(
            diff < 1e-4,
            "With temp=2, scaled should be raw/2, diff={diff}"
        );
    }

    #[test]
    fn test_parameter_count() {
        let device = Default::default();
        let model = EnergyHeadConfig::new(4096).init::<TestBackend>(&device);
        let count = model.num_params();

        // Layer 1: 4096*2048 (weight) + 2048 (bias) = 8,390,656
        // Layer 2: 2048*1024 (weight) + 1024 (bias) = 2,098,176
        // Layer 3: 1024*512 (weight) + 512 (bias) = 524,800
        // Layer 4: 512*1 (weight, no bias) = 512
        // log_temperature: 1
        // Total: 11,014,145
        assert_eq!(count, 11_014_145, "Expected 11,014,145 params, got {count}");
    }
}
