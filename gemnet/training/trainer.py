import numpy as np
import logging
import torch

from .schedules import LinearWarmupExponentialDecay
from .ema_decay import ExponentialMovingAverage

class Trainer:
    """
    Trainer for LMDB/PyG-based datasets.

    Parameters
    ----------
        model: Model
            Model to train.
        learning_rate: float
            Initial learning rate.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule.
        weight_decay: float
            Weight decay factor of the AdamW optimizer.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay
        grad_clip_max: float
            Gradient clipping threshold.
        decay_patience: int
            Learning rate decay on plateau. Number of evaluation intervals after decaying the learning rate.
        decay_factor: float
            Learning rate decay on plateau. Multiply inverse of decay factor by learning rate to obtain new learning rate.
        decay_cooldown: int
            Learning rate decay on plateau. Number of evaluation intervals after which to return to normal operation.
        ema_decay: float
            Decay to use to maintain the moving averages of trained variables.
        rho_force: float
            Weighting factor for the force loss compared to the energy. In range [0,1]
                loss = loss_energy * (1-rho_force) + loss_force * rho_force
        loss: str
            Name of the loss objective of the forces.
        mve: bool
            If True perform Mean Variance Estimation.
        agc: bool
            If True use adaptive gradient clipping else clip by global norm.
        device: torch.device or str
            Device to use for training ("cuda" or "cpu").
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        decay_steps: int = 100000,
        decay_rate: float = 0.96,
        warmup_steps: int = 0,
        weight_decay: float = 0.001,
        staircase: bool = False,
        grad_clip_max: float = 1000,
        decay_patience: int = 10,
        decay_factor: float = 0.5,
        decay_cooldown: int = 10,
        ema_decay: float = 0.999,
        rho_force: float = 0.99,
        loss: str = "mae",
        mve: bool = False,
        agc: bool = False,
        device=None,
    ):
        assert 0 <= rho_force <= 1

        self.model = model
        self.ema_decay = ema_decay
        self.grad_clip_max = grad_clip_max
        self.rho_force = float(rho_force)
        self.mve = mve
        self.loss = loss
        self.agc = agc

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        if mve:
            self.tracked_metrics = [
                "loss",
                "energy_mae",
                "energy_nll",
                "energy_var",
                "force_mae",
                "force_rmse",
                "force_nll",
                "force_var",
            ]
        else:
            self.tracked_metrics = ["loss", "energy_mae", "force_mae", "force_rmse"]

        self.reset_optimizer(
            learning_rate,
            weight_decay,
            warmup_steps,
            decay_steps,
            decay_rate,
            staircase,
            decay_patience,
            decay_factor,
            decay_cooldown,
        )

    def reset_optimizer(
        self,
        learning_rate,
        weight_decay,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase,
        decay_patience,
        decay_factor,
        decay_cooldown,
    ):
        if weight_decay > 0:
            adamW_params = []
            rest_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "atom_emb" in name or "frequencies" in name or "bias" in name:
                        rest_params.append(param)
                    else:
                        adamW_params.append(param)
            # AdamW optimizer
            AdamW = torch.optim.AdamW(
                adamW_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                weight_decay=weight_decay,
                amsgrad=True,
            )
            lr_schedule_AdamW = LinearWarmupExponentialDecay(
                AdamW, warmup_steps, decay_steps, decay_rate, staircase
            )
            # Adam: optimizer for embeddings, frequencies and biases
            Adam = torch.optim.Adam(
                rest_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                amsgrad=True,
            )
            lr_schedule_Adam = LinearWarmupExponentialDecay(
                Adam, warmup_steps, decay_steps, decay_rate, staircase
            )
            self.schedulers = MultiWrapper(lr_schedule_AdamW, lr_schedule_Adam)
            self.optimizers = MultiWrapper(AdamW, Adam)
        else:
            # Adam: optimizer for all parameters
            Adam = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                amsgrad=True,
            )
            lr_schedule_Adam = LinearWarmupExponentialDecay(
                Adam, warmup_steps, decay_steps, decay_rate, staircase
            )
            self.schedulers = MultiWrapper(lr_schedule_Adam)
            self.optimizers = MultiWrapper(Adam)

        # LR decay on plateau
        self.plateau_callback = ReduceLROnPlateau(
            optimizer=self.optimizers,
            scheduler=self.schedulers,
            factor=decay_factor,
            patience=decay_patience,
            cooldown=decay_cooldown,
            verbose=True,
        )

        if self.agc:
            self.params_except_last = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and ("out_energy" in name or "out_forces" in name):
                    self.params_except_last.append(param)

        self.exp_decay = ExponentialMovingAverage(
            [p for p in self.model.parameters() if p.requires_grad], self.ema_decay
        )

    def save_variable_backups(self):
        self.exp_decay.store()

    def load_averaged_variables(self):
        self.exp_decay.copy_to()

    def restore_variable_backups(self):
        self.exp_decay.restore()

    def decay_maybe(self, val_loss):
        self.plateau_callback.step(val_loss)

    @staticmethod
    def _unitwise_norm(x, norm_type=2.0):
        if x.ndim <= 1:
            return x.norm(norm_type)
        else:
            return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

    @staticmethod
    def _adaptive_gradient_clipping(parameters, clip_factor=0.05, eps=1e-3, norm_type=2.0):
        with torch.no_grad():
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            for p in parameters:
                if p.grad is None:
                    continue
                p_data = p
                g_data = p.grad
                max_norm = (
                    Trainer._unitwise_norm(p_data, norm_type=norm_type)
                    .clamp_(min=eps)
                    .mul_(clip_factor)
                )
                grad_norm = Trainer._unitwise_norm(g_data, norm_type=norm_type)
                clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
                new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
                p.grad.copy_(new_grads)

    def scale_shared_grads(self):
        with torch.no_grad():
            def scale_grad(param, scale_factor):
                if param.grad is None:
                    return
                param.grad.copy_(param.grad / scale_factor)

            shared_int_layers = [
                getattr(self.model, "mlp_rbf3", None),
                getattr(self.model, "mlp_cbf3", None),
                getattr(self.model, "mlp_rbf_h", None),
            ]
            shared_int_layers = [layer for layer in shared_int_layers if layer is not None]
            if not getattr(self.model, "triplets_only", False):
                shared_int_layers += [
                    getattr(self.model, "mlp_rbf4", None),
                    getattr(self.model, "mlp_cbf4", None),
                    getattr(self.model, "mlp_sbf4", None),
                ]
                shared_int_layers = [layer for layer in shared_int_layers if layer is not None]
            for layer in shared_int_layers:
                scale_grad(layer.weight, getattr(self.model, "num_blocks", 1))
            if hasattr(self.model, "mlp_rbf_out"):
                scale_grad(self.model.mlp_rbf_out.weight, getattr(self.model, "num_blocks", 1) + 1)

    def get_mae(self, targets, pred):
        return torch.nn.functional.l1_loss(pred, targets, reduction="mean")

    def get_rmse(self, targets, pred):
        return torch.mean(torch.norm((pred - targets), p=2, dim=1))

    def get_nll(self, targets, mean_pred, var_pred):
        return torch.nn.functional.gaussian_nll_loss(
            mean_pred, targets, var_pred, reduction="mean"
        )

    def predict(self, batch):
        # batch is a PyG Batch object
        energy, forces = self.model(batch)
        if self.mve:
            mean_energy = energy[:, :1]
            var_energy = torch.nn.functional.softplus(energy[:, 1:])
            mean_forces = forces[:, 0, :]
            var_forces = torch.nn.functional.softplus(forces[:, 1, :])
            return mean_energy, var_energy, mean_forces, var_forces
        else:
            if len(forces.shape) == 3:
                forces = forces[:, 0]
            return energy, None, forces, None

    def train_on_batch(self, dataset_iter, metrics):
        self.model.train()
        batch = next(dataset_iter)
        batch = batch.to(self.device)

        mean_energy, var_energy, mean_forces, var_forces = self.predict(batch)

        # Targets (must be present in Data objects)
        energy_target = getattr(batch, "y", None)
        force_target = getattr(batch, "forces", None)
        if energy_target is None:
            raise ValueError("Batch does not have attribute 'y' for energy target.")
        if force_target is None:
            raise ValueError("Batch does not have attribute 'forces' for force target.")

        if self.mve:
            energy_nll = self.get_nll(energy_target, mean_energy, var_energy)
            force_nll = self.get_nll(force_target, mean_forces, var_forces)
            loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll
        else:
            energy_mae = self.get_mae(energy_target, mean_energy)
            if self.loss == "mae":
                force_metric = self.get_mae(force_target, mean_forces)
            else:
                force_metric = self.get_rmse(force_target, mean_forces)
            loss = energy_mae * (1 - self.rho_force) + self.rho_force * force_metric

        self.optimizers.zero_grad()
        loss.backward()
        self.scale_shared_grads()

        if self.agc and hasattr(self, "params_except_last"):
            self._adaptive_gradient_clipping(
                self.params_except_last, clip_factor=self.grad_clip_max
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_max
            )

        self.optimizers.step()
        self.schedulers.step()
        self.exp_decay.update()

        loss = loss.detach()
        with torch.no_grad():
            if self.mve:
                energy_mae = self.get_mae(energy_target, mean_energy)
                force_mae = self.get_mae(force_target, mean_forces)
                force_rmse = self.get_rmse(force_target, mean_forces)
            else:
                if self.loss == "mae":
                    force_mae = force_metric
                    force_rmse = self.get_rmse(force_target, mean_forces)
                else:
                    force_mae = self.get_mae(force_target, mean_forces)
                    force_rmse = force_metric

            if self.mve:
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                    force_nll=force_nll,
                    force_var=var_forces,
                )
            else:
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                )

        return loss

    def test_on_batch(self, dataset_iter, metrics):
        self.model.eval()
        batch = next(dataset_iter)
        batch = batch.to(self.device)

        energy_target = getattr(batch, "y", None)
        force_target = getattr(batch, "forces", None)
        if energy_target is None:
            raise ValueError("Batch does not have attribute 'y' for energy target.")
        if force_target is None:
            raise ValueError("Batch does not have attribute 'forces' for force target.")

        if getattr(self.model, "direct_forces", False):
            with torch.no_grad():
                mean_energy, var_energy, mean_forces, var_forces = self.predict(batch)
        else:
            mean_energy, var_energy, mean_forces, var_forces = self.predict(batch)

        with torch.no_grad():
            energy_mae = self.get_mae(energy_target, mean_energy)
            force_mae = self.get_mae(force_target, mean_forces)
            force_rmse = self.get_rmse(force_target, mean_forces)
            if self.mve:
                energy_nll = self.get_nll(energy_target, mean_energy, var_energy)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_mae
                force_nll = self.get_nll(force_target, mean_forces, var_forces)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                    force_nll=force_nll,
                    force_var=var_forces,
                )
            else:
                force_metric = force_mae if self.loss == "mae" else force_rmse
                loss = (1 - self.rho_force) * energy_mae + self.rho_force * force_metric
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                )
        return loss

    def eval_on_batch(self, dataset_iter):
        self.model.eval()
        batch = next(dataset_iter)
        batch = batch.to(self.device)
        with torch.no_grad():
            energy, _, forces, _ = self.predict(batch)
        return (energy, forces), batch

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "model",
                "schedulers",
                "optimizers",
                "plateau_callback",
                "exp_decay",
            ]
        }
        for attr in ["schedulers", "optimizers", "plateau_callback", "exp_decay"]:
            state_dict.update({attr: getattr(self, attr).state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        trainer_dict = {
            key: value
            for key, value in self.state_dict().items()
            if key
            not in [
                "model",
                "schedulers",
                "optimizers",
                "plateau_callback",
                "exp_decay",
            ]
        }
        self.__dict__.update(trainer_dict)
        for attr in ["schedulers", "optimizers", "plateau_callback", "exp_decay"]:
            getattr(self, attr).load_state_dict(state_dict[attr])

class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        scheduler,
        factor=0.1,
        patience=10,
        threshold=1e-4,
        max_reduce=10,
        cooldown=0,
        threshold_mode="rel",
        min_lr=0,
        eps=1e-8,
        mode="min",
        verbose=False,
    ):
        if factor >= 1.0:
            raise ValueError(f"Factor should be < 1.0 but is {factor}.")
        self.factor = factor
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(optimizer, MultiWrapper):
            self.optimizer = optimizer.wrapped
        if isinstance(scheduler, MultiWrapper):
            self.scheduler = scheduler.wrapped

        if not isinstance(self.optimizer, (list, tuple)):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]

        assert len(self.optimizer) == len(self.scheduler)
        for opt in self.optimizer:
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"{type(opt).__name__} is not an Optimizer but is of type {type(opt)}")

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_steps = None
        self.mode_worse = None
        self.eps = eps
        self.last_step = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()
        self._reduce_counter = 0

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_steps = 0

    def step(self, metrics):
        current = float(metrics)
        step = self.last_step + 1
        self.last_step = step

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_steps = 0

        if self.num_bad_steps > self.patience:
            self._reduce(step)
            self.cooldown_counter = self.cooldown
            self.num_bad_steps = 0

    def _reduce(self, step):
        self._reduce_counter += 1
        for optimizer, schedule in zip(self.optimizer, self.scheduler):
            if hasattr(schedule, "base_lrs"):
                schedule.base_lrs = [lr * self.factor for lr in schedule.base_lrs]
            else:
                raise ValueError(
                    "Schedule does not have attribute 'base_lrs' for the learning rate."
                )
        if self.verbose:
            logging.info(f"Step {step}: reducing on plateu by {self.factor}.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = np.inf
        else:
            self.mode_worse = -np.inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "scheduler"]
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )

class MultiWrapper:
    def __init__(self, *ops):
        self.wrapped = ops

    def __getitem__(self, idx):
        return self.wrapped[idx]

    def zero_grad(self):
        for op in self.wrapped:
            op.zero_grad()

    def step(self):
        for op in self.wrapped:
            op.step()

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.wrapped)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.wrapped):
            opt.load_state_dict(state_dict[i])
