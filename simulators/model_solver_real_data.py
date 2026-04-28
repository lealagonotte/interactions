"""
Differentiable (Soft) Cellular Automaton for Wildfire Spread
Faithful differentiable relaxation of CellularAutomaton_humidity_age.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path



# 1 Soft / Differentiable CA

class SoftFireCA(nn.Module):
    """
    Differentiable relaxation of CellularAutomaton_humidity_age.

    Faithfully mirrors the true CA update:

        next[i,j] = state[i,j]
                  + age_factor[i,j] * psi(moisture[i,j])
                  * sum_{k∈N(i,j)} dist_coeff(k) * wind[k] * phi(delta_h_{k→i,j}) * state[k]

    then soft-clipped to [0,1] via clamp instead of np.clip.

    Parameters learned (stored as logs for positivity):
        alpha  — Peterson exponent in age inflammability
        beta   — moisture dampening strength  (psi = exp(-beta·m))
        gamma  — slope effect strength        (phi uphill/downhill)

    Fixed hyperparameters matching the true CA:
        t_max  = 30   (age saturation threshold)
        p_max  = 1.0  (max inflammability)

    NOTE: wind_grid is NOT stored in the model — it is passed at each
    forward() call so that each simulation can use its own wind field.
    """

    T_MAX = 30.0
    P_MAX = 1.0

    def __init__(self, height_grid, age_grid, moisture_grid,
                 alpha_init=1.0, beta_init=1.0, gamma_init=1.0,
                 burn_mask=None):
        super().__init__()

        # Learnable log-parameters
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha_init), dtype=torch.float32))
        self.log_beta  = nn.Parameter(torch.tensor(np.log(beta_init),  dtype=torch.float32))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init), dtype=torch.float32))

        def t(x):
            return torch.tensor(x, dtype=torch.float32)

        self.register_buffer("height",   t(height_grid))
        self.register_buffer("age",      t(age_grid))
        self.register_buffer("moisture", t(moisture_grid))

        # burn_mask : (H,W) in [0,1]  — 1 = can burn, 0 = fireproof
        # If None, all cells can burn (equivalent to all-ones mask)
        if burn_mask is not None:
            self.register_buffer("burn_mask", t(burn_mask))
        else:
            self.register_buffer("burn_mask", None)

        self.H, self.W = height_grid.shape

        # Diagonal / cardinal distance coefficients (matches true CA)
        # offsets and their dist_coeff: 0.83 if diagonal, 1.0 if cardinal
        self._offsets = [
            (-1,-1, 0.83), (-1, 0, 1.0), (-1, 1, 0.83),
            ( 0,-1, 1.0 ),               ( 0, 1, 1.0 ),
            ( 1,-1, 0.83), ( 1, 0, 1.0), ( 1, 1, 0.83),
        ]

    # parameter accessors
    @property
    def alpha(self): return torch.exp(self.log_alpha)

    @property
    def beta(self):  return torch.exp(self.log_beta)

    @property
    def gamma(self): return torch.exp(self.log_gamma)

    # differentiable functions
    def _age_inflammability(self):
        alpha = self.alpha
        ratio = self.age / self.T_MAX

        below = (1.0 + self.P_MAX) ** (ratio ** alpha) - 1.0
        above = torch.full_like(below, self.P_MAX)

        return torch.where(self.age < self.T_MAX, below, above)

    def _phi(self, dh):
        gamma = self.gamma

        downhill = torch.exp(gamma * dh)
        uphill = 1.0 + gamma * torch.sqrt(torch.clamp(dh, min=0.0))

        return torch.where(dh <= 0, downhill, uphill)

    def _psi(self):
        return torch.exp(-self.beta * self.moisture)   

    # one forward step
    def step(self, state, wind):
        """
        Mirrors CellularAutomaton_humidity_age.evolve() exactly:

            total_influence[i,j] = sum_k dist_coeff * wind[k] * phi(h[i,j]-h[k]) * state[k]
            next[i,j] = state[i,j] + age_factor[i,j] * psi[i,j] * total_influence[i,j]

        then soft-clipped to [0,1].

        state : (H, W) tensor
        wind  : (H, W) tensor — wind field specific to this simulation
        """
        age_factor = self._age_inflammability()   # (H,W)
        psi        = self._psi()                  # (H,W)

        total_influence = torch.zeros_like(state)

        for di, dj, dist_coeff in self._offsets:
            ni = torch.arange(self.H) + di
            nj = torch.arange(self.W) + dj

            valid_i = (ni >= 0) & (ni < self.H)
            valid_j = (nj >= 0) & (nj < self.W)

            ni = ni.clamp(0, self.H - 1)
            nj = nj.clamp(0, self.W - 1)

            s_nb = state[ni[:, None], nj[None, :]]
            w_nb = wind[ni[:, None], nj[None, :]]
            h_nb = self.height[ni[:, None], nj[None, :]]

            mask = valid_i[:, None] & valid_j[None, :]

            dh = self.height - h_nb
            phi = self._phi(dh)

            total_influence += dist_coeff * w_nb * phi * s_nb * mask

        next_state = state + age_factor * psi * total_influence

        # Soft clip to [0,1]
        next_state = torch.clamp(next_state, 0.0, 1.0)

        # Apply burn mask: fireproof cells (mask=0) are forced back to 0
        if self.burn_mask is not None:
            next_state = next_state * self.burn_mask

        return next_state

    # full rollout that gives the predicted arrival time 
    def forward(self, ignition_point, ignition_value, wind_grid, n_steps, n_substeps=1):
        """
        Runs the soft CA for n_steps observed timesteps, each divided into
        n_substeps internal CA steps. Arrival time is recorded at the coarse
        timestep level (every n_substeps CA steps).
        Returns predicted_arrival : (H, W)  in [0, n_steps]

        ignition_point : (i0, j0)
        wind_grid      : (H, W) array or tensor — wind specific to this fire
        n_steps        : number of observed timesteps
        n_substeps     : number of CA micro-steps per observed timestep (default 1)
        """
        i0, j0 = ignition_point
        state = torch.zeros(self.H, self.W)
        state[i0, j0] = torch.tensor(ignition_value, dtype=state.dtype)

        # Convert wind to tensor if needed
        if not isinstance(wind_grid, torch.Tensor):
            wind_grid = torch.tensor(wind_grid, dtype=torch.float32)

        prev_state = torch.zeros_like(state)
        arrival    = torch.full((self.H, self.W), float(n_steps))

        for t in range(1, n_steps + 1):
            # Run n_substeps internal CA steps before recording arrival
            for _ in range(n_substeps):
                state = self.step(state, wind_grid)
            # Arrival recorded at coarse timestep level
            first_ignition = torch.clamp(state - prev_state, min=0.0)
            arrival = arrival - first_ignition * (n_steps - t)
            prev_state = state.detach()   # detach to avoid O(T2) memory

        return arrival


# 2 Loss

def combined_loss(pred_arrival, obs_arrival, n_steps, lambda_unburned=0.5):
    pred_norm = pred_arrival / n_steps
    
    burned_mask = obs_arrival >= 0
    loss_burned = torch.mean((pred_norm[burned_mask] - obs_arrival[burned_mask].float() / n_steps) ** 2)
    
    unburned_mask = ~burned_mask
    loss_unburned = torch.mean((pred_norm[unburned_mask] - 1.0) ** 2)
    
    return loss_burned + lambda_unburned * loss_unburned

def masked_mse(pred_arrival, obs_arrival, n_steps):
    mask = obs_arrival >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0)
    pred_norm = pred_arrival[mask] / n_steps
    obs_norm  = obs_arrival[mask].float() / n_steps
    return torch.mean((pred_norm - obs_norm) ** 2)

# 3 Training loop

def fit_no_wind(model, fires_data, n_steps=100,n_substeps=1, n_epochs=150, lr=0.05, verbose=True):
    """
    fires_data : list of dicts with keys:
        'ignition_point' : (i0, j0)
        'ignition_value' : float between 0 and 1
        'arrival_time'   : np.ndarray (H,W) int, -1 = never burned
        'wind_grid'      : np.ndarray (H,W) — wind specific to this fire
    """
    optimizer = optim.Adam([model.log_alpha, model.log_beta, model.log_gamma], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-3)

    history = {"loss": [], "alpha": [], "beta": [], "gamma": []}

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for fire in fires_data:
            ig     = fire["ignition_point"]
            ig_val = torch.tensor(fire["ignition_value"], dtype=torch.float32)
            obs    = torch.tensor(fire["arrival_time"], dtype=torch.float32)
            wind   = torch.tensor(fire["wind_grid"],    dtype=torch.float32)

            pred = model(ig, ig_val, wind, n_steps, n_substeps)
            loss = masked_mse(pred, obs, n_steps)
            total_loss = total_loss + loss

        total_loss = total_loss / len(fires_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([model.log_alpha, model.log_beta, model.log_gamma], 1.0)
        optimizer.step()
        scheduler.step()

        a = model.alpha.item()
        b = model.beta.item()
        g = model.gamma.item()
        history["loss"].append(total_loss.item())
        history["alpha"].append(a)
        history["beta"].append(b)
        history["gamma"].append(g)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:4d}  loss={total_loss.item():.5f}  "
                  f"α={a:.3f}  β={b:.3f}  γ={g:.3f}")

    return history


# 4 Diagnostic plot

def plot_results_no_wind(history, true_alpha, true_beta, true_gamma):
    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    ax_loss  = fig.add_subplot(gs[0])
    ax_alpha = fig.add_subplot(gs[1])
    ax_beta  = fig.add_subplot(gs[2])
    ax_gamma = fig.add_subplot(gs[3])

    epochs = range(len(history["loss"]))
    palette = {"loss": "red", "alpha": "blue", "beta": "yellow", "gamma": "green"}

    for ax, key, true_val, label in [
        (ax_loss,  "loss",  None,       "Loss"),
        (ax_alpha, "alpha", true_alpha, "α (age)"),
        (ax_beta,  "beta",  true_beta,  "β (moisture)"),
        (ax_gamma, "gamma", true_gamma, "γ (slope)"),
    ]:
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_title(label, fontsize=10, pad=8)

        color = palette[key]
        ax.plot(epochs, history[key], color=color, lw=1.8)

        if true_val is not None:
            ax.axhline(true_val, lw=1.2, ls="--", alpha=0.6, label=f"true={true_val}")
            ax.legend(fontsize=7)

    fig.suptitle("Soft CA — Gradient-Based Parameter Recovery", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("recovery_plot.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("Plot saved to recovery_plot.png")










"""
Differentiable (Soft) Cellular Automaton for Wildfire Spread
Faithful differentiable relaxation of CellularAutomaton_humidity_age.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path



# 1 Soft / Differentiable CA

class SoftFireCA_wind(nn.Module):
    """
    Differentiable relaxation of CellularAutomaton_humidity_age.

    Faithfully mirrors the true CA update:

        next[i,j] = state[i,j]
                  + age_factor[i,j] * psi(moisture[i,j])
                  * sum_{k∈N(i,j)} dist_coeff(k) * wind_factor(k) * phi(delta_h_{k→i,j}) * state[k]

    then soft-clipped to [0,1] via clamp instead of np.clip.

    Parameters learned (stored as logs for positivity):
        alpha  — Peterson exponent in age inflammability
        beta   — moisture dampening strength  (psi = exp(-beta·m))
        gamma  — slope effect strength        (phi uphill/downhill)
        delta  — wind direction alignment strength
                 wind_factor = exp(delta * alignment)
                 where alignment = dot(propagation_unit_vec, wind_vec_at_neighbour)

    Fixed hyperparameters matching the true CA:
        t_max  = 30   (age saturation threshold)
        p_max  = 1.0  (max inflammability)

    NOTE: wind grids are NOT stored in the model — they are passed at each
    forward() call so that each simulation can use its own wind field.
    """

    T_MAX = 30.0
    P_MAX = 1.0

    def __init__(self, height_grid, age_grid, moisture_grid,
                 alpha_init=1.0, beta_init=1.0, gamma_init=1.0, delta_init=1.0,
                 burn_mask=None):
        super().__init__()

        # Learnable log-parameters
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha_init), dtype=torch.float32))
        self.log_beta  = nn.Parameter(torch.tensor(np.log(beta_init),  dtype=torch.float32))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init), dtype=torch.float32))
        self.log_delta = nn.Parameter(torch.tensor(np.log(delta_init), dtype=torch.float32))

        def t(x):
            return torch.tensor(x, dtype=torch.float32)

        self.register_buffer("height",   t(height_grid))
        self.register_buffer("age",      t(age_grid))
        self.register_buffer("moisture", t(moisture_grid))

        # burn_mask : (H,W) in [0,1]  — 1 = can burn, 0 = fireproof
        # If None, all cells can burn (equivalent to all-ones mask)
        if burn_mask is not None:
            self.register_buffer("burn_mask", t(burn_mask))
        else:
            self.register_buffer("burn_mask", None)

        self.H, self.W = height_grid.shape

        # Diagonal / cardinal distance coefficients (matches true CA)
        # offsets and their dist_coeff: 0.83 if diagonal, 1.0 if cardinal
        # Also precompute unit propagation vector (neighbour -> cell) for wind alignment
        self._offsets = []
        for di, dj in [(-1,-1), (-1,0), (-1,1),
                        (0,-1),          (0,1),
                        (1,-1),  (1,0),  (1,1)]:
            dist_coeff = 0.83 if abs(di) + abs(dj) == 2 else 1.0
            norm = np.sqrt(di**2 + dj**2)
            ui, uj = di / norm, dj / norm   # unit vector: neighbour -> current cell
            self._offsets.append((di, dj, dist_coeff, ui, uj))

    # parameter accessors
    @property
    def alpha(self): return torch.exp(self.log_alpha)

    @property
    def beta(self):  return torch.exp(self.log_beta)

    @property
    def gamma(self): return torch.exp(self.log_gamma)

    @property
    def delta(self): return torch.exp(self.log_delta)

    # differentiable functions
    def _age_inflammability(self):
        alpha = self.alpha
        ratio = torch.clamp(self.age / self.T_MAX, min=1e-6)

        below = (1.0 + self.P_MAX) ** (ratio ** alpha) - 1.0
        above = torch.full_like(below, self.P_MAX)

        return torch.where(self.age < self.T_MAX, below, above)

    def _phi(self, dh):
        gamma = self.gamma

        downhill = torch.exp(gamma * dh)
        uphill = 1.0 + gamma * torch.sqrt(torch.clamp(dh, min=0.0))

        return torch.where(dh <= 0, downhill, uphill)

    def _psi(self):
        return torch.exp(-self.beta * self.moisture)

    # one forward step
    def step(self, state, wind_speed, wind_x, wind_y):
        """
        state      : (H, W) tensor
        wind_speed : (H, W) tensor — wind speed magnitude
        wind_x     : (H, W) tensor — eastward wind component  (speed * sin(dir))
        wind_y     : (H, W) tensor — northward wind component (speed * cos(dir))

        wind_factor for each neighbour k -> cell (i,j):
            alignment   = dot(propagation_unit_vec, wind_vec_at_k)
            wind_factor = exp(delta * alignment)
        """
        age_factor = self._age_inflammability()   # (H,W)
        psi        = self._psi()                  # (H,W)
        delta      = self.delta

        total_influence = torch.zeros_like(state)

        for di, dj, dist_coeff, ui, uj in self._offsets:
            ni = torch.arange(self.H) + di
            nj = torch.arange(self.W) + dj

            valid_i = (ni >= 0) & (ni < self.H)
            valid_j = (nj >= 0) & (nj < self.W)

            ni = ni.clamp(0, self.H - 1)
            nj = nj.clamp(0, self.W - 1)

            s_nb  = state[ni[:, None], nj[None, :]]
            wx_nb = wind_x[ni[:, None], nj[None, :]]
            wy_nb = wind_y[ni[:, None], nj[None, :]]
            h_nb  = self.height[ni[:, None], nj[None, :]]

            mask = valid_i[:, None] & valid_j[None, :]

            # Topographic effect
            dh  = self.height - h_nb
            phi = self._phi(dh)

            alignment   = ui * wy_nb + uj * wx_nb   # (H,W)
            wind_factor = torch.exp(delta * alignment).clamp(-2,2)

            total_influence += dist_coeff * wind_factor * phi * s_nb * mask

        next_state = state + age_factor * psi * total_influence

        # Soft clip to [0,1]
        next_state = torch.clamp(next_state, 0.0, 1.0)

        # Apply burn mask: fireproof cells (mask=0) are forced back to 0
        if self.burn_mask is not None:
            next_state = next_state * self.burn_mask

        return next_state

    # full rollout that gives the predicted arrival time
    def forward(self, ignition_point, ignition_value, wind_speed, wind_dir, n_steps, n_substeps=1):
        """
        Runs the soft CA for n_steps observed timesteps, each divided into
        n_substeps internal CA steps. Arrival time is recorded at the coarse
        timestep level (every n_substeps CA steps).
        Returns predicted_arrival : (H, W) in [0, n_steps]

        ignition_point : (i0, j0)
        ignition_value : float in [0, 1] — initial burn intensity
        wind_speed     : (H, W) array or tensor — wind speed magnitude
        wind_dir       : (H, W) array or tensor — wind direction in radians
                         (meteorological convention: 0=North, pi/2=East)
                         NOTE: pass np.deg2rad(degrees + 180) if your data is
                         in degrees and follows the "direction from" convention.
        n_steps        : number of observed timesteps
        n_substeps     : number of CA micro-steps per observed timestep (default 1)
        """
        i0, j0 = ignition_point
        state = torch.zeros(self.H, self.W)
        state[i0, j0] = torch.tensor(ignition_value, dtype=state.dtype)

        # Convert to tensors if needed
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                return torch.tensor(x, dtype=torch.float32)
            return x

        wind_speed = to_tensor(wind_speed)
        wind_dir   = to_tensor(wind_dir)

        # Decompose wind into cartesian components
        wind_x = wind_speed * torch.sin(wind_dir)   # eastward
        wind_y = wind_speed * torch.cos(wind_dir)   # northward

        prev_state = torch.zeros_like(state)
        arrival    = torch.full((self.H, self.W), float(n_steps))

        for t in range(1, n_steps + 1):
            # Run n_substeps internal CA steps before recording arrival
            for _ in range(n_substeps):
                state = self.step(state, wind_speed, wind_x, wind_y)
            # Arrival recorded at coarse timestep level
            first_ignition = torch.clamp(state - prev_state, min=0.0)
            arrival = arrival - first_ignition * (n_steps - t)
            prev_state = state.detach()   # detach to avoid O(T^2) memory

        return arrival


# 2 Loss

def combined_loss(pred_arrival, obs_arrival, n_steps, lambda_unburned=0.5):
    pred_norm = pred_arrival / n_steps

    burned_mask = obs_arrival >= 0
    loss_burned = torch.mean((pred_norm[burned_mask] - obs_arrival[burned_mask].float() / n_steps) ** 2)

    unburned_mask = ~burned_mask
    loss_unburned = torch.mean((pred_norm[unburned_mask] - 1.0) ** 2)

    return loss_burned + lambda_unburned * loss_unburned

def masked_mse(pred_arrival, obs_arrival, n_steps):
    mask = obs_arrival >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0)
    pred_norm = pred_arrival[mask] / n_steps
    obs_norm  = obs_arrival[mask].float() / n_steps
    return torch.mean((pred_norm - obs_norm) ** 2)


# 3 Training loop

def fit_wind(model, fires_data, n_steps=100, n_substeps=1, n_epochs=150, lr=0.05, verbose=True):
    """
    fires_data : list of dicts with keys:
        'ignition_point' : (i0, j0)
        'ignition_value' : float between 0 and 1
        'arrival_time'   : np.ndarray (H,W) int, -1 = never burned
        'wind_speed'     : np.ndarray (H,W) — wind speed magnitude
        'wind_dir'       : np.ndarray (H,W) — wind direction in radians
    """
    params = [model.log_alpha, model.log_beta, model.log_gamma, model.log_delta]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-3)

    history = {"loss": [], "alpha": [], "beta": [], "gamma": [], "delta": []}

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for fire in fires_data:
            ig     = fire["ignition_point"]
            ig_val = fire["ignition_value"]
            obs    = torch.tensor(fire["arrival_time"], dtype=torch.float32)
            wspd   = torch.tensor(fire["wind_speed"],   dtype=torch.float32)
            wdir   = torch.tensor(fire["wind_dir"],     dtype=torch.float32)

            pred = model(ig, ig_val, wspd, wdir, n_steps, n_substeps)
            loss = masked_mse(pred, obs, n_steps)
            total_loss = total_loss + loss

        total_loss = total_loss / len(fires_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        a = model.alpha.item()
        b = model.beta.item()
        g = model.gamma.item()
        d = model.delta.item()
        history["loss"].append(total_loss.item())
        history["alpha"].append(a)
        history["beta"].append(b)
        history["gamma"].append(g)
        history["delta"].append(d)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:4d}  loss={total_loss.item():.5f}  "
                  f"α={a:.3f}  β={b:.3f}  γ={g:.3f}  δ={d:.3f}")

    return history


# 4 Diagnostic plot

def plot_results_wind(history, true_alpha, true_beta, true_gamma, true_delta=None):
    fig = plt.figure(figsize=(18, 4))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    axes   = [fig.add_subplot(gs[i]) for i in range(5)]
    epochs = range(len(history["loss"]))
    palette = {"loss": "red", "alpha": "blue", "beta": "orange", "gamma": "green", "delta": "purple"}

    for ax, key, true_val, label in [
        (axes[0], "loss",  None,        "Loss"),
        (axes[1], "alpha", true_alpha,  "α (age)"),
        (axes[2], "beta",  true_beta,   "β (moisture)"),
        (axes[3], "gamma", true_gamma,  "γ (slope)"),
        (axes[4], "delta", true_delta,  "δ (wind dir)"),
    ]:
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_title(label, fontsize=10, pad=8)
        ax.plot(epochs, history[key], color=palette[key], lw=1.8)
        if true_val is not None:
            ax.axhline(true_val, lw=1.2, ls="--", alpha=0.6, label=f"true={true_val}")
            ax.legend(fontsize=7)

    fig.suptitle("Soft CA — Gradient-Based Parameter Recovery", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("recovery_plot.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("Plot saved to recovery_plot.png")