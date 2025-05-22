from dataclasses import dataclass
from math import sqrt
from typing import Any, Generic, Self, cast

import torch
from einops import rearrange, reduce
from torch import nn, relu

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.activations import ACTIVATIONS_MAP
from crosscode.models.activations.topk import BatchTopkActivation
from crosscode.models.base_crosscoder import BaseCrosscoder, TActivation
from crosscode.models.initialization.init_strategy import InitStrategy
from crosscode.saveable_module import DTYPE_TO_STRING, STRING_TO_DTYPE
from crosscode.trainers.trainer import ModelWrapper
from crosscode.utils import calculate_vector_norm_fvu_X, get_fvu_dict, l1_norm


class ModelHookpointAcausalCrosscoder(Generic[TActivation], BaseCrosscoder[TActivation]):
    def __init__(
        self,
        n_models: int,
        n_hookpoints: int,
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
        init_strategy: InitStrategy["ModelHookpointAcausalCrosscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        crosscoding_dims = (n_models, n_hookpoints)

        super().__init__(
            crosscoding_dims,
            d_model,
            crosscoding_dims,
            d_model,
            n_latents,
            activation_fn,
            use_encoder_bias,
            use_decoder_bias,
            None,
            dtype,
        )
        self._crosscoding_dims = crosscoding_dims

        self.n_models = n_models
        self.n_hookpoints = n_hookpoints
        self.d_model = d_model

        if init_strategy is not None:
            init_strategy.init_weights(self)

    @dataclass
    class ForwardResult:
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        recon_acts_BMPD: torch.Tensor

    def forward_train(self, activation_BMPD: torch.Tensor) -> ForwardResult:
        res = self._forward_train(activation_BMPD)
        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            recon_acts_BMPD=res.output_BXoDo,
        )

    def forward(self, activation_BMPD: torch.Tensor) -> torch.Tensor:
        return self.forward_train(activation_BMPD).recon_acts_BMPD

    def decode_BMPD(self, latents_BL: torch.Tensor) -> torch.Tensor:
        return self.decode_BXoDo(latents_BL)

    @property
    def W_dec_LMPD(self) -> nn.Parameter:
        return self._W_dec_LXoDo

    @property
    def W_enc_MPDL(self) -> nn.Parameter:
        return self._W_enc_XiDiL

    @property
    def b_dec_MPD(self) -> nn.Parameter | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "n_hookpoints": self.n_hookpoints,
            "d_model": self.d_model,
            "n_latents": self.n_latents,
            "activation_fn": {
                "classname": self.activation_fn.__class__.__name__,
                "cfg": self.activation_fn._dump_cfg(),
            },
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": self._b_dec_XoDo is not None,
            "dtype": DTYPE_TO_STRING[self._dtype],
        }

    @classmethod
    def _scaffold_from_cfg(cls: type[Self], cfg: dict[str, Any]):
        activation = cfg["activation_fn"]
        activation_fn_cls = ACTIVATIONS_MAP[activation["classname"]]
        activation_fn = cast(TActivation, activation_fn_cls._scaffold_from_cfg(activation["cfg"]))

        return ModelHookpointAcausalCrosscoder(
            n_models=cfg["n_models"],
            n_hookpoints=cfg["n_hookpoints"],
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=STRING_TO_DTYPE[cfg["dtype"]],
        )

    def fold_activation_scaling_into_weights_(self, scaling_factors_out_MP: torch.Tensor) -> None:
        self._fold_activation_scaling_into_weights_(scaling_factors_out_MP, scaling_factors_out_MP)

    def with_folded_scaling_factors(self, scaling_factors_out_MP: torch.Tensor) -> Self:
        return self._with_folded_scaling_factors(scaling_factors_out_MP, scaling_factors_out_MP)


class RichReLUTranscoder(Generic[TActivation], nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_latents: int,
        # latent_activation_fn: TActivation,
        k: int,
        ref_W_in: torch.Tensor,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_latents = n_latents

        # self.latent_activation_fn = latent_activation_fn
        self.k = k
        self.mlp_W_up_DH = nn.Parameter(ref_W_in.clone())

        self.sparse_enc_HL = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty((d_hidden, n_latents))))
        self.sparse_dec_LD = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty((n_latents, d_model))))


    def topk_activation(self, x_BL: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values_BK, indices_BK = x_BL.topk(self.k, dim=-1, sorted=False)
        out_BL = torch.zeros_like(x_BL)
        out_BL.scatter_(-1, indices_BK, values_BK)
        return out_BL, indices_BK

    @dataclass
    class ForwardResult:
        ff_hidden_BH: torch.Tensor
        latent_pre_act_BL: torch.Tensor
        latent_acts_BL: torch.Tensor
        recon_acts_BD: torch.Tensor
        indices_used_BK: torch.Tensor

    def forward_train(self, in_act_BD: torch.Tensor) -> ForwardResult:
        ff_hidden_BH = relu(in_act_BD @ self.mlp_W_up_DH)

        latent_pre_act_BL = ff_hidden_BH @ self.sparse_enc_HL
        latent_acts_BL, indices_BK = self.topk_activation(latent_pre_act_BL)

        recon_acts_BD = latent_acts_BL @ self.sparse_dec_LD

        return self.ForwardResult(
            ff_hidden_BH=ff_hidden_BH,
            latent_pre_act_BL=latent_pre_act_BL,
            latent_acts_BL=latent_acts_BL,
            recon_acts_BD=recon_acts_BD,
            indices_used_BK=indices_BK,
        )


# class RichSwiGLUTranscoder(Generic[TActivation], nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         d_hidden: int,
#         n_latents: int,
#         k: int,
#     ):
#         super().__init__()

#         self.d_model = d_model
#         self.d_hidden = d_hidden
#         self.n_latents = n_latents

#         self.latent_activation_fn = BatchTopkActivation(k_per_example=k)

#         self.mlp_W_up_DH = nn.Parameter(torch.empty((d_model, d_hidden)))
#         self.mlp_W_gate_DH = nn.Parameter(torch.empty((d_model, d_hidden)))
#         self.sparse_enc_HL = nn.Parameter(torch.empty((d_hidden, n_latents)))
#         self.sparse_dec_LD = nn.Parameter(torch.empty((n_latents, d_model)))

#     @dataclass
#     class ForwardResult:
#         ff_hidden_BH: torch.Tensor
#         latent_pre_act_BL: torch.Tensor
#         latent_acts_BL: torch.Tensor
#         recon_acts_BD: torch.Tensor
#         indices_used_BK: torch.Tensor

#     def forward_train(self, in_act_BD: torch.Tensor) -> ForwardResult:
#         x_BH = in_act_BD @ self.mlp_W_up_DH
#         gate_BH = in_act_BD @ self.mlp_W_gate_DH
#         ff_hidden_BH = x_BH * gate_BH

#         latent_pre_act_BL = ff_hidden_BH @ self.sparse_enc_HL
#         latent_acts_BL = self.latent_activation_fn.forward(latent_pre_act_BL)

#         recon_acts_BD = latent_acts_BL @ self.sparse_dec_LD

#         return self.ForwardResult(
#             ff_hidden_BH=ff_hidden_BH,
#             latent_pre_act_BL=latent_pre_act_BL,
#             latent_acts_BL=latent_acts_BL,
#             recon_acts_BD=recon_acts_BD,
#             indices_used_BK=indices_used_BK,
#         )


class RichTranscoderWrapper(ModelWrapper):
    def __init__(self, model: RichReLUTranscoder[TActivation]): #  | RichSwiGLUTranscoder[TActivation]):
        self.model = model

    def run_batch(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        acts = batch.activations_BMPD.to(self.model.mlp_W_up_DH.device)
        _, n_models, n_hookpoints, _ = acts.shape
        assert n_models == 1
        assert n_hookpoints == 2

        in_act_BD = acts[:, 0, 0, :]
        target_out_BD = acts[:, 0, 1, :]

        res = self.model.forward_train(in_act_BD)

        recon_loss = (res.recon_acts_BD - target_out_BD).square().mean()
        recon_loss /= self.model.d_model ** 2

        fvu = calculate_vector_norm_fvu_X(res.recon_acts_BD, target_out_BD)

        # indices_used_BL = res.indices_used_BL
        l1_norms_L = reduce(self.model.sparse_enc_HL, "hidden latent -> latent", l1_norm)
        rank_loss = l1_norms_L[res.indices_used_BK].mean()
        # rank_loss = rank_loss_L.mean() * 0.1

        loss = recon_loss + rank_loss

        if log:
            logs = {
                "recon_loss": recon_loss.item(),
                "rank_loss": rank_loss.item(),
                "loss": loss.item(),
                "fvu": fvu.item(),
            }
            return loss, logs

        return loss, None

    def save(self, step: int):
        ...

    def expensive_logs(self) -> dict[str, Any]:
        return {}

    def parameters(self):
        return self.model.parameters()
