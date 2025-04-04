from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from crosscode.data.activations_dataloader import ModelHookpointActivationsDataloader
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.activation_function import ActivationFunction
from crosscode.trainers.base_acausal_trainer import BaseModelHookpointAcausalTrainer
from crosscode.trainers.config_common import BaseTrainConfig
from crosscode.trainers.utils import create_cosine_sim_and_relative_norm_histograms_diffing

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseFebUpdateDiffingTrainer(Generic[TConfig, TAct], BaseModelHookpointAcausalTrainer[TConfig, TAct]):
    activations_dataloader: ModelHookpointActivationsDataloader

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: ModelHookpointActivationsDataloader,
        model: ModelHookpointAcausalCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        n_shared_latents: int,
    ):
        super().__init__(cfg, activations_dataloader, model, wandb_run, device, save_dir)
        self.n_shared_latents = n_shared_latents
        assert self.model.n_models == 2, "expected the model crosscoding dim to have length 2"
        assert self.activations_dataloader.n_models == 2, "expected the activations dataloader to have length 2"

    def _after_forward_passes(self):
        self._synchronise_shared_weight_grads()

    def _synchronise_shared_weight_grads(self) -> None:
        assert self.model.W_dec_LMPD.grad is not None
        W_dec_grad_LMPD = self.model.W_dec_LMPD.grad[: self.n_shared_latents]

        model_0_grad_LPD = W_dec_grad_LMPD[:, 0]
        model_1_grad_LPD = W_dec_grad_LMPD[:, 1]

        summed_grad = model_0_grad_LPD + model_1_grad_LPD
        model_0_grad_LPD.copy_(summed_grad)
        model_1_grad_LPD.copy_(summed_grad)

        m0_grads, m1_grads = self.model.W_dec_LMPD.grad[: self.n_shared_latents].unbind(dim=1)
        assert (m0_grads == m1_grads).all()

        m0_weights, m1_weights = self.model.W_dec_LMPD[: self.n_shared_latents].unbind(dim=1)
        assert (m0_weights == m1_weights).all()

    def _step_logs(self) -> dict[str, Any]:
        log_dict = super()._step_logs()

        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            W_dec_LiMPD = self.model.W_dec_LMPD[self.n_shared_latents :].detach()
            for p, hookpoint_name in enumerate(self.activations_dataloader.hookpoints):
                W_dec_LiMD = W_dec_LiMPD[:, :, p]
                relative_decoder_norms_plot, shared_features_cosine_sims_plot = (
                    create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_LMD=W_dec_LiMD)
                )
                if relative_decoder_norms_plot is not None:
                    log_dict.update(
                        {
                            f"{hookpoint_name}_relative_decoder_norms": relative_decoder_norms_plot,
                        }
                    )
                if shared_features_cosine_sims_plot is not None:
                    log_dict.update(
                        {
                            f"{hookpoint_name}_shared_features_cosine_sims": shared_features_cosine_sims_plot,
                        }
                    )

        return log_dict
