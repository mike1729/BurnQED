use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;

/// Configuration for a spectral-normalized linear layer.
#[derive(Config, Debug)]
pub struct SpectralNormLinearConfig {
    /// Input dimension.
    pub d_input: usize,
    /// Output dimension.
    pub d_output: usize,
    /// Number of power iterations for spectral norm estimation.
    #[config(default = 5)]
    pub n_power_iterations: usize,
    /// Small constant for numerical stability in normalization.
    #[config(default = 1e-12)]
    pub eps: f64,
    /// Whether to include a bias term.
    #[config(default = true)]
    pub bias: bool,
}

/// Linear layer with spectral normalization (Option C: random reinit per forward).
///
/// Weight is normalized by its spectral norm (largest singular value)
/// at each forward pass using power iteration from fresh random vectors.
/// This constrains the layer's Lipschitz constant to 1.
#[derive(Module, Debug)]
pub struct SpectralNormLinear<B: Backend> {
    /// Weight matrix, shape (d_output, d_input). Kaiming initialized.
    pub(crate) weight: Param<Tensor<B, 2>>,
    /// Optional bias, shape (d_output,). Zero initialized.
    bias: Option<Param<Tensor<B, 1>>>,
    /// Number of power iterations per forward pass.
    n_power_iterations: usize,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl SpectralNormLinearConfig {
    /// Initialize a SpectralNormLinear layer.
    ///
    /// Weight uses Kaiming (He) initialization: Normal(0, sqrt(2/fan_in)).
    /// Bias is zero-initialized.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralNormLinear<B> {
        let std_dev = (2.0 / self.d_input as f64).sqrt();
        let weight = Tensor::random(
            [self.d_output, self.d_input],
            Distribution::Normal(0.0, std_dev),
            device,
        );

        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.d_output], device)))
        } else {
            None
        };

        SpectralNormLinear {
            weight: Param::from_tensor(weight),
            bias,
            n_power_iterations: self.n_power_iterations,
            eps: self.eps,
        }
    }
}

impl<B: Backend> SpectralNormLinear<B> {
    /// Forward pass with spectral normalization.
    ///
    /// 1. Estimates spectral norm via power iteration with fresh random u,v
    /// 2. Normalizes weight by spectral norm: W_normed = W / sigma
    /// 3. Computes output = input @ W_normed^T + bias
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = self.weight.val(); // (d_out, d_in), on autodiff graph
        let [d_out, d_in] = weight.dims();
        let device = input.device();

        // Fresh random vectors each forward (Option C: random reinit)
        let mut u: Tensor<B, 1> =
            Tensor::random([d_out], Distribution::Normal(0.0, 1.0), &device);
        let mut v: Tensor<B, 1> =
            Tensor::random([d_in], Distribution::Normal(0.0, 1.0), &device);

        // Power iteration to estimate largest singular value
        for _ in 0..self.n_power_iterations {
            // v = W^T @ u / ||W^T @ u||
            // (d_in, d_out) @ (d_out, 1) = (d_in, 1) -> squeeze -> (d_in,)
            let wt_u: Tensor<B, 1> = weight
                .clone()
                .transpose()
                .matmul(u.clone().unsqueeze_dim::<2>(1))
                .squeeze::<1>(1);
            let wt_u_norm = wt_u.clone().powf_scalar(2.0).sum().sqrt() + self.eps;
            v = wt_u / wt_u_norm;

            // u = W @ v / ||W @ v||
            // (d_out, d_in) @ (d_in, 1) = (d_out, 1) -> squeeze -> (d_out,)
            let w_v: Tensor<B, 1> = weight
                .clone()
                .matmul(v.clone().unsqueeze_dim::<2>(1))
                .squeeze::<1>(1);
            let w_v_norm = w_v.clone().powf_scalar(2.0).sum().sqrt() + self.eps;
            u = w_v / w_v_norm;
        }

        // sigma = u^T @ W @ v (scalar)
        // (1, d_out) @ (d_out, d_in) @ (d_in, 1) = (1, 1) -> reshape to (1,)
        let sigma: Tensor<B, 1> = u
            .unsqueeze_dim::<2>(0) // (1, d_out)
            .matmul(
                weight
                    .clone()
                    .matmul(v.unsqueeze_dim::<2>(1)), // W @ v: (d_out, 1)
            ) // (1, 1)
            .squeeze::<1>(1) // (1,)
            .abs(); // ensure positive

        // W_normed = W / sigma  (division is on autodiff graph)
        let w_normed = weight / sigma.unsqueeze_dim::<2>(1); // broadcast (1,1) over (d_out, d_in)

        // output = input @ W_normed^T + bias
        let output = input.matmul(w_normed.transpose());
        match &self.bias {
            Some(b) => {
                let bias_val: Tensor<B, 1> = b.val();
                output + bias_val.unsqueeze_dim::<2>(0)
            }
            None => output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_output_shape() {
        let device = Default::default();
        let layer = SpectralNormLinearConfig::new(64, 32).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::random(
            [4, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        assert_eq!(output.dims(), [4, 32]);
    }

    #[test]
    fn test_spectral_norm_bounds() {
        let device = Default::default();
        let layer = SpectralNormLinearConfig::new(64, 32).init::<TestBackend>(&device);

        let probe = Tensor::<TestBackend, 2>::random(
            [100, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(probe.clone());

        // Check that no input vector gets amplified by more than ~1.2x
        let input_norms = probe.powf_scalar(2.0).sum_dim(1).sqrt();
        let output_norms = output.powf_scalar(2.0).sum_dim(1).sqrt();
        let ratios = output_norms / input_norms;
        let max_ratio: f32 = ratios.max().into_scalar().elem();

        assert!(
            max_ratio < 1.2,
            "Max ratio {max_ratio} exceeds 1.2 — spectral norm not bounded"
        );
    }

    #[test]
    fn test_no_bias() {
        let device = Default::default();
        let layer = SpectralNormLinearConfig::new(16, 8)
            .with_bias(false)
            .init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [2, 16],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        assert_eq!(output.dims(), [2, 8]);

        // Zero input should give zero output with no bias
        let zero_input = Tensor::<TestBackend, 2>::zeros([2, 16], &device);
        let zero_output = layer.forward(zero_input);
        let max_val: f32 = zero_output.abs().max().into_scalar().elem();
        assert!(
            max_val < 1e-6,
            "No-bias layer should map zero input to zero, got max {max_val}"
        );
    }

    #[test]
    fn test_deterministic_weight() {
        let device = Default::default();
        // Use more power iterations for tighter convergence
        let layer = SpectralNormLinearConfig::new(32, 16)
            .with_n_power_iterations(20)
            .init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [4, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Two forward passes with same weight, different random u/v
        let output1 = layer.forward(input.clone());
        let output2 = layer.forward(input);

        // Should be nearly identical — power iteration converges
        let diff = (output1 - output2).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();
        assert!(
            max_diff < 0.05,
            "Two forwards differ by {max_diff}, expected < 0.05"
        );
    }

    #[test]
    fn test_gradient_flows() {
        use burn::optim::GradientsParams;

        let device = Default::default();
        let layer = SpectralNormLinearConfig::new(16, 8).init::<TestAutodiffBackend>(&device);

        let input = Tensor::<TestAutodiffBackend, 2>::random(
            [2, 16],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        let loss = output.sum();

        let grads = GradientsParams::from_grads(loss.backward(), &layer);
        let weight_grad = grads
            .get::<NdArray<f32>, 2>(layer.weight.id)
            .expect("weight should have gradient");

        let grad_sum: f32 = weight_grad.abs().sum().into_scalar().elem();
        assert!(
            grad_sum > 0.0,
            "Weight gradient is all zeros — gradient not flowing through spectral norm"
        );
    }
}
