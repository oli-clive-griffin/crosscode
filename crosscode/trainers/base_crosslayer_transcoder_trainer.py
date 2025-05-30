from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch, ModelHookpointActivationsDataloader
from crosscode.models.activations.activation_function import ActivationFunction
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.trainers.base_acausal_trainer import BaseTrainer
from crosscode.trainers.config_common import BaseTrainConfig
from crosscode.trainers.wandb_utils.main import create_checkpoint_artifact
from crosscode.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseCrossLayerTranscoderTrainer(
    Generic[TConfig, TAct],
    BaseTrainer[TConfig, CrossLayerTranscoder[TAct], ModelHookpointActivationsBatch],
):
    activations_dataloader: ModelHookpointActivationsDataloader

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: ModelHookpointActivationsDataloader,
        model: CrossLayerTranscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        out_hookpoints: list[str],
    ):
        super().__init__(cfg, activations_dataloader, model, wandb_run, device, save_dir)
        self.out_hookpoints = out_hookpoints
        assert self.activations_dataloader.n_models == 1
        assert self.activations_dataloader.n_hookpoints == len(self.out_hookpoints) + 1

    def run_batch(
        self, batch: ModelHookpointActivationsBatch, log: bool
    ) -> tuple[torch.Tensor, dict[str, float] | None, int]:
        batch_BMPD = batch.activations_BMPD
        assert batch_BMPD.shape[1] == 1, "we must have one model"
        assert batch_BMPD.shape[2] == len(self.out_hookpoints) + 1, "we must have one more hookpoint than out layers"

        in_BD = batch_BMPD[:, 0, 0]
        target_BPD = batch_BMPD[:, 0, 1:]

        train_res = self.model.forward_train(in_BD)

        self.firing_tracker.add_batch(train_res.latents_BL)

        loss, log_dict = self._calculate_loss_and_log(train_res, target_BPD, log=log)
        return loss, log_dict, batch_BMPD.shape[0]

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        train_res: CrossLayerTranscoder.ForwardResult,
        target_BPD: torch.Tensor,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def _get_fvu_dict(self, y_BPD: torch.Tensor, recon_y_BPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            y_BPD,
            recon_y_BPD,
            ("hookpoint", self.out_hookpoints),
        )

    def _maybe_save_model(self) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            # We know this is (model, hookpoint) because it's a ModelHookpointActivationsDataloader
            scaling_factors_MP = self.activations_dataloader.get_scaling_factors()
            assert scaling_factors_MP.shape[0] == 1, "expected the scaling factors to have one model only"
            scaling_factors_P = scaling_factors_MP[0]
            self.model.with_folded_scaling_factors(scaling_factors_P).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)
