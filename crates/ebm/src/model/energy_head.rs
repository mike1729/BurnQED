use burn::module::Param;
use burn::prelude::*;

use crate::model::spectral_norm::{SpectralNormLinear, SpectralNormLinearConfig};

/// Configuration for the EnergyHead MLP.
///
/// Maps encoder hidden states (d_encoder) to scalar energy via a 3-layer
/// spectral-normed SiLU MLP with dropout and learnable temperature.
///
/// ```text
/// (batch, d_encoder)
///   → SpectralNormLinear(d_encoder→d_hidden1) → SiLU → Dropout
///   → SpectralNormLinear(d_hidden1→d_hidden2) → SiLU → Dropout
///   → SpectralNormLinear(d_hidden2→1, no bias) → squeeze
///   → raw_energy / exp(log_temperature)
///   → energy: (batch,)
/// ```
#[derive(Config, Debug)]
pub struct EnergyHeadConfig {
    /// Encoder output dimension (e.g. 4096 for DeepSeek-Prover-V2-7B).
    pub d_encoder: usize,
    /// First hidden layer dimension.
    #[config(default = 512)]
    pub d_hidden1: usize,
    /// Second hidden layer dimension.
    #[config(default = 256)]
    pub d_hidden2: usize,
    /// Dropout probability applied after each SiLU activation.
    #[config(default = 0.1)]
    pub dropout: f64,
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
    /// Output spectral-normed linear: d_hidden2 → 1 (no bias).
    sn_linear3: SpectralNormLinear<B>,
    /// Dropout after first SiLU.
    dropout1: burn::nn::Dropout,
    /// Dropout after second SiLU.
    dropout2: burn::nn::Dropout,
    /// Log-temperature parameter. Energy is divided by exp(log_temperature).
    /// Initialized to 0 so initial temperature = 1.
    log_temperature: Param<Tensor<B, 1>>,
}

impl EnergyHeadConfig {
    /// Initialize an EnergyHead with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> EnergyHead<B> {
        EnergyHead {
            sn_linear1: SpectralNormLinearConfig::new(self.d_encoder, self.d_hidden1).init(device),
            sn_linear2: SpectralNormLinearConfig::new(self.d_hidden1, self.d_hidden2).init(device),
            sn_linear3: SpectralNormLinearConfig::new(self.d_hidden2, 1)
                .with_bias(false)
                .init(device),
            dropout1: burn::nn::DropoutConfig::new(self.dropout).init(),
            dropout2: burn::nn::DropoutConfig::new(self.dropout).init(),
            log_temperature: Param::from_tensor(Tensor::zeros([1], device)),
        }
    }
}

impl<B: Backend> EnergyHead<B> {
    /// Forward pass: maps encoder hidden states to scalar energy values.
    ///
    /// Input shape: `(batch, d_encoder)`
    /// Output shape: `(batch,)`
    pub fn forward(&self, h: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = self.sn_linear1.forward(h);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout1.forward(x);

        let x = self.sn_linear2.forward(x);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout2.forward(x);

        let raw_energy: Tensor<B, 1> = self.sn_linear3.forward(x).squeeze::<1>(1);
        let temperature = self.log_temperature.val().exp();
        raw_energy / temperature
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

        // Check gradient flows to log_temperature
        let temp_grad = grads
            .get::<NdArray<f32>, 1>(model.log_temperature.id)
            .expect("log_temperature should have gradient");
        let temp_grad_sum: f32 = temp_grad.abs().sum().into_scalar().elem();
        assert!(
            temp_grad_sum > 0.0,
            "log_temperature gradient is zero — gradient not flowing"
        );
    }

    #[test]
    fn test_temperature_scaling() {
        // Temperature scaling test: verify that changing log_temperature affects
        // the output magnitude. We use more power iterations (20) to reduce
        // spectral norm variance between forward calls.
        let device = Default::default();

        let sn_config = |d_in: usize, d_out: usize, bias: bool| {
            crate::model::spectral_norm::SpectralNormLinearConfig::new(d_in, d_out)
                .with_n_power_iterations(20)
                .with_bias(bias)
        };

        // Build model with high power iterations for tighter convergence
        let model = EnergyHead {
            sn_linear1: sn_config(32, 16, true).init::<TestBackend>(&device),
            sn_linear2: sn_config(16, 8, true).init::<TestBackend>(&device),
            sn_linear3: sn_config(8, 1, false).init::<TestBackend>(&device),
            dropout1: burn::nn::DropoutConfig::new(0.0).init(),
            dropout2: burn::nn::DropoutConfig::new(0.0).init(),
            log_temperature: Param::from_tensor(Tensor::zeros([1], &device)),
        };

        let input = Tensor::<TestBackend, 2>::random(
            [32, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Default log_temperature=0 → temperature=1
        let energy_temp1 = model.forward(input.clone());

        // Create model with log_temperature = ln(2) → temperature = 2
        let model_temp2 = EnergyHead {
            sn_linear1: model.sn_linear1,
            sn_linear2: model.sn_linear2,
            sn_linear3: model.sn_linear3,
            dropout1: model.dropout1,
            dropout2: model.dropout2,
            log_temperature: Param::from_tensor(Tensor::from_floats(
                [2.0_f32.ln()],
                &device,
            )),
        };
        let energy_temp2 = model_temp2.forward(input);

        // energy_temp1 / energy_temp2 ≈ 2.0 (dividing by 2x temperature halves energy)
        // With 20 power iterations and batch=32, spectral norm variance is much lower.
        // Use element-wise absolute energy comparison: |e1| should be roughly 2x |e2|.
        let abs1: f32 = energy_temp1.abs().mean().into_scalar().elem();
        let abs2: f32 = energy_temp2.abs().mean().into_scalar().elem();
        let ratio = abs1 / abs2;
        assert!(
            (ratio - 2.0).abs() < 1.0,
            "Expected |energy1|/|energy2| ratio ~2.0, got {ratio} (abs1={abs1}, abs2={abs2})"
        );
    }

    #[test]
    fn test_parameter_count() {
        let device = Default::default();
        let model = EnergyHeadConfig::new(4096).init::<TestBackend>(&device);
        let count = model.num_params();

        // Layer 1: 4096*512 (weight) + 512 (bias) = 2,097,664
        // Layer 2: 512*256 (weight) + 256 (bias) = 131,328
        // Layer 3: 256*1 (weight, no bias) = 256
        // log_temperature: 1
        // Total: 2,229,249
        assert_eq!(count, 2_229_249, "Expected 2,229,249 params, got {count}");
    }
}
