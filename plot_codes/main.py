import numpy as np
from scipy.linalg import sqrtm

class LinearFineTuningExperiment:
    def __init__(self, dx, dy, true_rank, noise_std=0.1,
                 sv_decay_type=None, sv_decay_rate=None, sv_scale=1.0):

        self.dx = dx
        self.dy = dy
        self.true_rank = true_rank
        self.noise_std = noise_std


        self.A_0 = np.random.randn(dy, dx)


        # Generate Delta_star with optional singular value control
        if sv_decay_type is None:
            # Default behavior: random U and V with scaling
            U = np.random.randn(dy, true_rank)
            V = np.random.randn(true_rank, dx)
            # Scale down to make it a 'fine-tuning' shift (perturbation)
            self.Delta_star = (U @ V) / np.sqrt(true_rank)
        else:
            # Controlled singular values with orthonormal U and V
            U, _ = np.linalg.qr(np.random.randn(dy, true_rank))
            V, _ = np.linalg.qr(np.random.randn(dx, true_rank))
            S = self._generate_singular_values(true_rank, sv_decay_type, sv_decay_rate, sv_scale)
            self.Delta_star = U @ np.diag(S) @ V.T

        self.A_star = self.A_0 + self.Delta_star

    def _generate_singular_values(self, rank, decay_type, decay_rate, scale):
        """
        Generate singular values based on specified decay pattern.

        Args:
            rank: Number of singular values to generate
            decay_type: Type of decay ("constant", "fast_decay", "slow_decay")
            decay_rate: Decay rate parameter (used for exponential and power-law decay)
            scale: Overall scale factor for singular values

        Returns:
            Array of singular values
        """
        if decay_type == "constant":
            return scale * np.ones(rank)
        elif decay_type == "fast_decay":
            # Exponential decay: scale * exp(-decay_rate * i)
            if decay_rate is None:
                decay_rate = 0.5
            return scale * np.exp(-decay_rate * np.arange(rank))
        elif decay_type == "slow_decay":
            # Power-law decay: scale / (i+1)^decay_rate
            if decay_rate is None:
                decay_rate = 0.2
            return scale / ((np.arange(rank) + 1) ** decay_rate)
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}. "
                           f"Expected 'constant', 'fast_decay', or 'slow_decay'.")

    def generate_dataset(self, n_samples):

        # Inputs x_i ~ N(0, I)
        X = np.random.randn(self.dx, n_samples)
        
        # Noise epsilon ~ N(0, noise_std^2 * I)
        Epsilon = np.random.randn(self.dy, n_samples) * self.noise_std
        
        # Labels y_i = A* x_i + epsilon
        Y = (self.A_star @ X) + Epsilon
        
        return X, Y

    def compute_excess_risk(self, A_hat):

        excess_risk = np.linalg.norm(A_hat - self.A_star, 'fro') ** 2
        return excess_risk

    def solve_fft(self, X_train, Y_train):
        """
        Implements Full Fine-Tuning (FFT) solution via Eq. (4).
        A_FFT = A_0 + (Y - A_0 X) X^dagger
        """
        # Residuals: Y - A_0 X
        Residual = Y_train - (self.A_0 @ X_train)
        X_pinv = np.linalg.pinv(X_train)
        A_fft = self.A_0 + (Residual @ X_pinv)
        
        return A_fft

    def solve_lora(self, X_train, Y_train, r_lora):

        n = X_train.shape[1]
        
        Sigma_hat_xx = (1/n) * (X_train @ X_train.T)
    
        U_cov, S_cov, Vt_cov = np.linalg.svd(Sigma_hat_xx, full_matrices=False)
        S_cov = np.maximum(S_cov, 0) 
        Sigma_sqrt = U_cov @ np.diag(np.sqrt(S_cov)) @ Vt_cov
        
        Sigma_sqrt_pinv = np.linalg.pinv(Sigma_sqrt)

        Residual = Y_train - (self.A_0 @ X_train)
        X_pinv = np.linalg.pinv(X_train)
        
        Target = Residual @ X_pinv @ Sigma_sqrt
        
 
        U_t, S_t, Vt_t = np.linalg.svd(Target, full_matrices=False)
        
        if r_lora < len(S_t):
            U_r = U_t[:, :r_lora]
            S_r = np.diag(S_t[:r_lora])
            Vt_r = Vt_t[:r_lora, :]
            Target_trunc = U_r @ S_r @ Vt_r
        else:
            Target_trunc = U_t @ np.diag(S_t) @ Vt_t
            

        D_hat = Target_trunc @ Sigma_sqrt_pinv
        
        A_lora = self.A_0 + D_hat
        
        return A_lora

# --- Main Execution Block ---

def run_experiment():
    print("--- Setting up Multivariate Regression Fine-Tuning ---")
    
 
    dx = 70
    dy = 40
    
    true_delta_rank = 1    # The "true" rank of the shift
    noise_level = 5      # eps noise
    

    exp = LinearFineTuningExperiment(dx, dy, true_delta_rank, noise_level)

    lora_r = 1

    print(f"Dimensions: dx={dx}, dy={dy}")
    print(f"True Delta Rank: {true_delta_rank}, LoRA Train Rank: {lora_r}")
    print("-" * 60)

    # --- Scenario A: Underdetermined Regime (n < dx) ---
    n_under = 30  # Less than dx=50
    print(f"\n[Scenario A] Underdetermined Regime (n={n_under} < dx={dx})")

    X_train_u, Y_train_u = exp.generate_dataset(n_under)

    # Solve FFT
    A_fft_u = exp.solve_fft(X_train_u, Y_train_u)
    err_fft_u = exp.compute_excess_risk(A_fft_u)

    # Solve LoRA
    A_lora_u = exp.solve_lora(X_train_u, Y_train_u, lora_r)
    err_lora_u = exp.compute_excess_risk(A_lora_u)

    print(f"FFT Excess Risk:  {err_fft_u:.5f}")
    print(f"LoRA Excess Risk: {err_lora_u:.5f}")
    if err_lora_u < err_fft_u:
        print(">> Result: LoRA outperformed FFT in underdetermined regime.")
    else:
        print(">> Result: FFT outperformed LoRA.")

    # --- Scenario B: Overdetermined Regime (n > dx) ---
    n_over = 200  # More than dx=50
    print(f"\n[Scenario B] Overdetermined Regime (n={n_over} > dx={dx})")

    X_train_o, Y_train_o = exp.generate_dataset(n_over)

    # Solve FFT
    A_fft_o = exp.solve_fft(X_train_o, Y_train_o)
    err_fft_o = exp.compute_excess_risk(A_fft_o)

    # Solve LoRA
    A_lora_o = exp.solve_lora(X_train_o, Y_train_o, lora_r)
    err_lora_o = exp.compute_excess_risk(A_lora_o)

    print(f"FFT Excess Risk:  {err_fft_o:.5f}")
    print(f"LoRA Excess Risk: {err_lora_o:.5f}")

    print("-" * 60)
  

if __name__ == "__main__":
    run_experiment()