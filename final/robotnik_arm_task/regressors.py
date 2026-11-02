#!/usr/bin/env python3
"""Registry-based model interface for behaviour-cloning regressors.

Each model class implements:
  - fit(X_train, y_train, X_val, y_val, **kwargs) -> metrics dict
  - save_checkpoint(path, **normalization_metadata) -> .pt file
  - load_model(cls, ckpt, device) -> callable model

Adding a new model is one decorated class:
    @register("gradient_boosting")
    class GBRegressor(BaseRegressor):
        ...
"""

import pickle
import time as _time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from activations import DEFAULT_ACTIVATION, LEGACY_ACTIVATION, get_activation


def build_mlp(input_dim, output_dim, hidden_dim, num_layers, activation=DEFAULT_ACTIVATION):
    layers = []
    prev = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(get_activation(activation))
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY = {}


def register(name):
    """Decorator to register a regressor class under a string key."""
    def decorator(cls):
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return decorator


def get_regressor(name):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown regressor '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def available_regressors():
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Sklearn wrapper (duck-types a torch model)
# ---------------------------------------------------------------------------
class SklearnModelWrapper:
    """Wraps a sklearn model to duck-type a torch model for predict()/forward()."""

    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model

    def __call__(self, x_tensor):
        if isinstance(x_tensor, torch.Tensor):
            x_np = x_tensor.cpu().numpy()
        else:
            x_np = np.asarray(x_tensor)
        pred = self.sklearn_model.predict(x_np)
        return torch.tensor(pred, dtype=torch.float32)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseRegressor(ABC):
    """Interface every regressor must implement."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train the model.

        Returns:
            dict with at least: val_mse, mae, p90
        """

    @abstractmethod
    def save_checkpoint(self, path, **meta):
        """Persist model + normalization metadata to a .pt file
        loadable by run_policy.load_policy()."""

    @classmethod
    @abstractmethod
    def load_model(cls, ckpt, device):
        """Reconstruct a callable model from a checkpoint dict.

        Returns:
            A callable (torch Module or wrapper) ready for inference.
        """


# ---------------------------------------------------------------------------
# MLP (wraps existing training loop)
# ---------------------------------------------------------------------------
@register("mlp")
class MLPRegressor(BaseRegressor):
    def __init__(self):
        self.model = None

    @classmethod
    def load_model(cls, ckpt, device):
        model = build_mlp(ckpt["input_dim"], ckpt["output_dim"],
                          ckpt["hidden_dim"], ckpt["num_layers"],
                          ckpt.get("activation", LEGACY_ACTIVATION))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train on MAE (L1), keep the checkpoint with the best validation MSE.

        Defaults follow the selected settings of the 618-run hyperparameter study.
        Optional kwargs:
          loss_weights: per-output weights for the training loss (default: all 1)
          on_epoch:     callback(epoch, train_loss, val_loss, val_pred, improved, model)
                        called after every epoch; val_pred is in normalised units
        """
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 2048)
        lr = kwargs.get("lr", 1e-3)
        hidden_dim = kwargs.get("hidden_dim", 512)
        num_layers = kwargs.get("num_layers", 3)
        activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        weight_decay = kwargs.get("weight_decay", 0.0)
        grad_clip = kwargs.get("grad_clip", 1.0)
        loss_weights = kwargs.get("loss_weights", None)
        early_stop = kwargs.get("early_stop", 0.0)
        patience = kwargs.get("patience", 20)
        num_workers = kwargs.get("num_workers", 0)
        device_name = kwargs.get("device", "cpu")
        on_epoch = kwargs.get("on_epoch", None)
        device = torch.device(device_name)
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = build_mlp(input_dim, output_dim, hidden_dim, num_layers, activation).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        if loss_weights is None:
            loss_weights = np.ones(output_dim)
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32, device=device)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32))
        loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                             pin_memory=(device.type != "cpu"),
                             persistent_workers=(num_workers > 0))
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, **loader_kwargs)

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        t_start = _time.time()

        try:
            for epoch in range(1, epochs + 1):
                # Train: weighted MAE, gradient clipping
                model.train()
                train_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = ((pred - yb).abs() * loss_weights).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    train_loss += loss.item() * xb.size(0)
                train_loss /= len(train_ds)

                # Validation: MSE
                model.eval()
                val_loss = 0.0
                val_pred = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                        val_loss += criterion(pred, yb).item() * xb.size(0)
                        if on_epoch is not None:
                            val_pred.append(pred.cpu().numpy())
                val_loss /= len(val_ds)

                if val_loss < best_val - early_stop:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if on_epoch is not None:
                    on_epoch(epoch, train_loss, val_loss, np.concatenate(val_pred), improved, model)
                elif epoch % max(1, epochs // 5) == 0 or epoch == 1:
                    print(f"  [mlp] epoch {epoch}/{epochs}  val_loss={val_loss:.6f}  best={best_val:.6f}")

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping: no improvement > {early_stop} for {patience} epochs")
                    break
        except KeyboardInterrupt:
            print(f"\nInterrupted at epoch {epoch}. Saving best checkpoint (epoch {best_epoch})...")

        elapsed = _time.time() - t_start

        # Restore best weights and compute final metrics
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = 0
        model.load_state_dict(best_state)
        model.eval()
        model.to(device)

        # Re-create val loader without workers (original may be dead after interrupt)
        val_loader_final = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
        all_pred = []
        with torch.no_grad():
            for xb, yb in val_loader_final:
                xb = xb.to(device)
                all_pred.append(model(xb).cpu().numpy())
        all_pred = np.concatenate(all_pred)

        abs_err = np.abs(all_pred - y_val)
        metrics = {
            "val_mse": float(best_val),
            "mae": float(abs_err.mean()),
            "p90": float(np.percentile(abs_err, 90)),
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
        }

        self.model = model
        self._best_state = best_state
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._activation = activation
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._best_val = best_val
        self._best_epoch = best_epoch
        self._elapsed = elapsed
        self._kwargs = kwargs

        return metrics

    def save_checkpoint(self, path, **meta):
        torch.save({
            "model_type": "mlp",
            "model_state_dict": self._best_state,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "activation": self._activation,
            "best_val_loss": self._best_val,
            "best_epoch": self._best_epoch,
            "elapsed_sec": self._elapsed,
            **meta,
        }, path)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class TransformerModel(nn.Module):
    """Input projection → N transformer encoder layers → output projection.

    Each input is treated as a single token (sequence length=1), so
    self-attention acts as a learned nonlinear mixing layer.
    """

    def __init__(self, input_dim, output_dim, d_model=512, nhead=8,
                 num_layers=4, dim_feedforward=2048, activation=DEFAULT_ACTIVATION):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # PyTorch's fast path only recognises relu/gelu when given by name
        act = activation if activation in ("relu", "gelu") else get_activation(activation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1, activation=act)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (B, input_dim) → treat as (B, 1, d_model) single-token sequence
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x)
        return self.output_proj(x.squeeze(1))


@register("transformer")
class TransformerRegressor(BaseRegressor):
    def __init__(self):
        self.model = None

    @classmethod
    def load_model(cls, ckpt, device):
        model = TransformerModel(
            input_dim=ckpt["input_dim"],
            output_dim=ckpt["output_dim"],
            d_model=ckpt["d_model"],
            nhead=ckpt["nhead"],
            num_layers=ckpt["num_layers"],
            dim_feedforward=ckpt["dim_feedforward"],
            activation=ckpt.get("activation", LEGACY_ACTIVATION),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 4096)
        lr = kwargs.get("lr", 1e-3)
        hidden_dim = kwargs.get("hidden_dim", 512)
        nhead = kwargs.get("nhead", 8)
        num_layers = kwargs.get("num_layers", 4)
        dim_feedforward = kwargs.get("dim_feedforward", hidden_dim * 4)
        activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        early_stop = kwargs.get("early_stop", 0.001)
        patience = kwargs.get("patience", 5)
        num_workers = kwargs.get("num_workers", 0)
        device_name = kwargs.get("device", "cpu")

        device = torch.device(device_name)
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = TransformerModel(
            input_dim, output_dim,
            d_model=hidden_dim, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            activation=activation,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32))
        loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                             pin_memory=(device.type != "cpu"),
                             persistent_workers=(num_workers > 0))
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, **loader_kwargs)

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        t_start = _time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_ds)

            if val_loss < best_val - early_stop:
                epochs_without_improvement = 0
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if val_loss >= best_val - early_stop:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  [transformer] Early stop at epoch {epoch}")
                break

            if epoch % max(1, epochs // 5) == 0 or epoch == 1:
                print(f"  [transformer] epoch {epoch}/{epochs}  val_loss={val_loss:.6f}  best={best_val:.6f}")

        elapsed = _time.time() - t_start

        # Restore best weights and compute final metrics
        model.load_state_dict(best_state)
        model.eval()
        model.to(device)

        all_pred = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                all_pred.append(model(xb).cpu().numpy())
        all_pred = np.concatenate(all_pred)

        abs_err = np.abs(all_pred - y_val)
        metrics = {
            "val_mse": float(best_val),
            "mae": float(abs_err.mean()),
            "p90": float(np.percentile(abs_err, 90)),
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
        }

        self.model = model
        self._best_state = best_state
        self._d_model = hidden_dim
        self._nhead = nhead
        self._num_layers = num_layers
        self._dim_feedforward = dim_feedforward
        self._activation = activation
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._best_val = best_val
        self._best_epoch = best_epoch
        self._elapsed = elapsed
        self._kwargs = kwargs

        return metrics

    def save_checkpoint(self, path, **meta):
        torch.save({
            "model_type": "transformer",
            "model_state_dict": self._best_state,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "d_model": self._d_model,
            "nhead": self._nhead,
            "num_layers": self._num_layers,
            "dim_feedforward": self._dim_feedforward,
            "activation": self._activation,
            "best_val_loss": self._best_val,
            "best_epoch": self._best_epoch,
            "elapsed_sec": self._elapsed,
            **meta,
        }, path)


# ---------------------------------------------------------------------------
# Sklearn-based regressors
# ---------------------------------------------------------------------------
class _SklearnRegressor(BaseRegressor):
    """Base for sklearn models. Subclasses set self.model in fit()."""

    @classmethod
    def load_model(cls, ckpt, device):
        sklearn_model = pickle.loads(ckpt["sklearn_model_bytes"])
        return SklearnModelWrapper(sklearn_model)

    def _compute_metrics(self, X_val, y_val):
        pred = self.model.predict(X_val)
        abs_err = np.abs(pred - y_val)
        mse = float(np.mean((pred - y_val) ** 2))
        return {
            "val_mse": mse,
            "mae": float(abs_err.mean()),
            "p90": float(np.percentile(abs_err, 90)),
        }

    def save_checkpoint(self, path, **meta):
        torch.save({
            "model_type": "sklearn",
            "sklearn_model_bytes": pickle.dumps(self.model),
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            **meta,
        }, path)


@register("linear")
class LinearRegressor(_SklearnRegressor):
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        from sklearn.linear_model import Ridge
        alpha = kwargs.get("alpha", 1.0)
        print(f"  [linear] Fitting Ridge(alpha={alpha}) on {X_train.shape[0]} samples...")
        t0 = _time.time()
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_train, y_train)
        elapsed = _time.time() - t0
        self._input_dim = X_train.shape[1]
        self._output_dim = y_train.shape[1]
        metrics = self._compute_metrics(X_val, y_val)
        metrics["elapsed_sec"] = elapsed
        metrics["best_epoch"] = 1
        print(f"  [linear] Done. val_mse={metrics['val_mse']:.6f}  mae={metrics['mae']:.4f}")
        return metrics


@register("random_forest")
class RandomForestRegressor(_SklearnRegressor):
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        from sklearn.ensemble import RandomForestRegressor as SkRF
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", None)
        n_jobs = kwargs.get("n_jobs", -1)
        max_samples_rf = kwargs.get("max_samples_rf", 50000)

        # Subsample for tractability — RF on 1M rows is very slow
        if X_train.shape[0] > max_samples_rf:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_train.shape[0], max_samples_rf, replace=False)
            X_fit, y_fit = X_train[idx], y_train[idx]
            print(f"  [random_forest] Subsampled {X_train.shape[0]} -> {max_samples_rf} for training")
        else:
            X_fit, y_fit = X_train, y_train

        print(f"  [random_forest] Fitting RF(n_estimators={n_estimators}, max_depth={max_depth}) "
              f"on {X_fit.shape[0]} samples...")
        t0 = _time.time()
        self.model = SkRF(n_estimators=n_estimators, max_depth=max_depth,
                          n_jobs=n_jobs, random_state=42)
        self.model.fit(X_fit, y_fit)
        elapsed = _time.time() - t0
        self._input_dim = X_train.shape[1]
        self._output_dim = y_train.shape[1]
        metrics = self._compute_metrics(X_val, y_val)
        metrics["elapsed_sec"] = elapsed
        metrics["best_epoch"] = 1
        print(f"  [random_forest] Done. val_mse={metrics['val_mse']:.6f}  mae={metrics['mae']:.4f}")
        return metrics


# ---------------------------------------------------------------------------
# Checkpoint loading dispatcher
# ---------------------------------------------------------------------------
_MODEL_TYPE_TO_CLASS = {
    "mlp": MLPRegressor,
    "transformer": TransformerRegressor,
    "sklearn": _SklearnRegressor,
}


def load_model_from_checkpoint(ckpt, device):
    """Reconstruct a callable model from a checkpoint dict.

    Reads ckpt["model_type"] and dispatches to the correct class's load_model.
    """
    model_type = ckpt.get("model_type", "mlp")
    cls = _MODEL_TYPE_TO_CLASS.get(model_type)
    if cls is None:
        raise KeyError(f"Unknown model_type '{model_type}' in checkpoint. "
                       f"Known: {list(_MODEL_TYPE_TO_CLASS.keys())}")
    return cls.load_model(ckpt, device)
