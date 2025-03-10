from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch as t
from torch import nn
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.token_hookpoint_dataloader import BaseTokenHookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.scripts.base_trainer import TConfig, save_model, validate_num_steps_per_epoch
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact

TAct = TypeVar("TAct", bound=ActivationFunction)


class BiTokenCCWrapper(nn.Module, Generic[TAct]):
    def __init__(
        self,
        single_token_cc: AcausalCrosscoder[TAct],
        double_token_cc: AcausalCrosscoder[TAct],
    ):
        super().__init__()

        assert single_token_cc.crosscoding_dims[0] == 1  # token
        assert len(single_token_cc.crosscoding_dims) == 2  # (token, hookpoint)
        self.single_cc = single_token_cc

        assert double_token_cc.crosscoding_dims[0] == 2  # token
        assert len(double_token_cc.crosscoding_dims) == 2  # (token, hookpoint)
        self.double_cc = double_token_cc

    @dataclass
    class TrainResult:
        recon_B1PD_single1: t.Tensor
        recon_B1PD_single2: t.Tensor
        recon_B2PD_double: t.Tensor
        hidden_BH_single1: t.Tensor
        hidden_BH_single2: t.Tensor
        hidden_BH_double: t.Tensor

    def forward_train(self, x_BTPD: t.Tensor) -> TrainResult:
        assert x_BTPD.shape[1] == 2

        output_single1 = self.single_cc.forward_train(x_BTPD[:, 0][:, None])
        output_single2 = self.single_cc.forward_train(x_BTPD[:, 1][:, None])
        output_both = self.double_cc.forward_train(x_BTPD)

        return self.TrainResult(
            recon_B1PD_single1=output_single1.recon_acts_BXD,
            recon_B1PD_single2=output_single2.recon_acts_BXD,
            recon_B2PD_double=output_both.recon_acts_BXD,
            hidden_BH_single1=output_single1.hidden_BH,
            hidden_BH_single2=output_single2.hidden_BH,
            hidden_BH_double=output_both.hidden_BH,
        )

    # stub forward for appeasing the nn.Module interface, but we don't use it
    def forward(self, x_BTPD: t.Tensor) -> t.Tensor:
        raise NotImplementedError("This method should not be called")


class BaseSlidingWindowCrosscoderTrainer(Generic[TAct, TConfig], ABC):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseTokenHookpointActivationsDataloader,
        crosscoders: BiTokenCCWrapper[TAct],
        wandb_run: Run,
        device: t.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.hookpoints = hookpoints

        self.crosscoders = crosscoders

        self.optimizer = build_optimizer(cfg.optimizer, self.crosscoders.parameters())

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader.num_batches()
        )

        self.total_steps = self.num_steps_per_epoch * (cfg.epochs or 1)
        logger.info(
            f"Total steps: {self.total_steps} (num_steps_per_epoch: {self.num_steps_per_epoch}, epochs: {cfg.epochs})"
        )

        self.lr_scheduler = (
            build_lr_scheduler(cfg.optimizer, self.total_steps) if cfg.optimizer.type == "adam" else None
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(
            activation_size=self.crosscoders.single_cc.hidden_dim
            + self.crosscoders.double_cc.hidden_dim
            + self.crosscoders.single_cc.hidden_dim
        )

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        scaling_factors_TP = self.activations_dataloader.get_norm_scaling_factors_TP()
        scaling_factor_1P = scaling_factors_TP.mean(dim=0, keepdim=True)

        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            for batch_BTPD in tqdm(
                islice(self.activations_dataloader.get_activations_iterator_BTPD(), self.num_steps_per_epoch),
                desc="Epoch Train Steps",
                total=self.num_steps_per_epoch,
            ):
                batch_BTPD = batch_BTPD.to(self.device)

                self._train_step(batch_BTPD)

                if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                    step_dir_single = self.save_dir / f"epoch_{self.epoch}_step_{self.step}_single"
                    step_dir_double = self.save_dir / f"epoch_{self.epoch}_step_{self.step}_double"

                    with self.crosscoders.single_cc.temporarily_fold_activation_scaling(scaling_factor_1P):
                        save_model(self.crosscoders.single_cc, step_dir_single)

                    with self.crosscoders.double_cc.temporarily_fold_activation_scaling(scaling_factors_TP):
                        save_model(self.crosscoders.double_cc, step_dir_double)

                    if self.cfg.upload_saves_to_wandb:
                        artifact = create_checkpoint_artifact(step_dir_single, self.wandb_run.id, self.step, self.epoch)
                        self.wandb_run.log_artifact(artifact)

                        artifact = create_checkpoint_artifact(step_dir_double, self.wandb_run.id, self.step, self.epoch)
                        self.wandb_run.log_artifact(artifact)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BTPD.shape[0]

                self.step += 1
            self.epoch += 1

    @abstractmethod
    def _train_step(self, batch_BTPD: t.Tensor) -> None: ...

    def _common_logs(self) -> dict[str, Any]:
        return {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }
