import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
from einops import rearrange
from transformer_lens import HookedTransformer  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import TokenSequenceLoader, build_tokens_sequence_loader
from model_diffing.log import logger
from model_diffing.scripts.config_common import DataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X
from model_diffing.utils import change_batch_size_BX


class BaseModelHookpointActivationsDataloader(ABC):
    @abstractmethod
    def get_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_MP(self) -> torch.Tensor: ...


class DummyModelHookpointActivationsDataloader(BaseModelHookpointActivationsDataloader):
    def __init__(self, batch_size: int, n_models: int, n_hookpoints: int, d_model: int, device: torch.device):
        self._batch_size = batch_size
        self._n_models = n_models
        self._n_hookpoints = n_hookpoints
        self._d_model = d_model
        self._device = device

    def get_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.randn(self._batch_size, self._n_models, self._n_hookpoints, self._d_model, device=self._device)

    def num_batches(self) -> int | None:
        return None

    def get_norm_scaling_factors_MP(self) -> torch.Tensor:
        return torch.ones(self._n_models, self._n_hookpoints)


class ScaledModelHookpointActivationsDataloader(BaseModelHookpointActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int | None,
        yield_batch_size: int,
        n_tokens_for_norm_estimate: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size

        norm_scaling_factors_MP = estimate_norm_scaling_factor_X(
            self._activations_iterator_BMPD(),  # don't pass the scaling factors here (becuase we're computing them!)
            n_tokens_for_norm_estimate,
        )

        self._norm_scaling_factors_MP = norm_scaling_factors_MP
        self._iterator = self._activations_iterator_BMPD(norm_scaling_factors_MP)

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def get_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]:
        return self._iterator

    def get_norm_scaling_factors_MP(self) -> torch.Tensor:
        return self._norm_scaling_factors_MP

    @torch.no_grad()
    def _activations_iterator_HsMPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_HSMPD = self._activations_harvester.get_activations_HSMPD(seq.tokens_HS)
            activations_HsMPD = rearrange(activations_HSMPD, "h s m p d -> (h s) m p d")
            special_tokens_mask_Hs = rearrange(seq.special_tokens_mask_HS, "h s -> (h s)")
            yield activations_HsMPD[~special_tokens_mask_Hs]

    @torch.no_grad()
    def _activations_iterator_BMPD(self, scaling_factors_MP: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        iterator_HsMPD = self._activations_iterator_HsMPD()

        device = next(iterator_HsMPD).device

        if scaling_factors_MP is None:
            scaling_factors_MP1 = torch.ones((1, 1, 1), device=device)
        else:
            scaling_factors_MP1 = rearrange(scaling_factors_MP, "m p -> m p 1").to(device)

        for batch_HsMPD in iterator_HsMPD:
            n_batches = batch_HsMPD.shape[0] // self._yield_batch_size
            for i in range(n_batches):
                batch_BMPD = batch_HsMPD[i * self._yield_batch_size : (i + 1) * self._yield_batch_size]
                yield batch_BMPD * scaling_factors_MP1

        # for batch_BMPD in change_batch_size_BX(iterator_HX=iterator_HsMPD, B=self._yield_batch_size):
        #     assert batch_BMPD.shape[0] == self._yield_batch_size, (
        #         f"batch_BMPD.shape[0] {batch_BMPD.shape[0]} != self._yield_batch_size {self._yield_batch_size}"
        #     )  # REMOVE ME
        #     yield batch_BMPD * scaling_factors_MP1


def build_dataloader(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
) -> ScaledModelHookpointActivationsDataloader:
    tokenizer = llms[0].tokenizer
    assert all(
        llm.tokenizer.special_tokens_map == tokenizer.special_tokens_map  # type: ignore
        for llm in llms
    ), "all tokenizers should have the same special tokens"
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.token_sequence_loader,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # Create activations cache directory if cache is enabled
    activations_cache_dir = None
    if cfg.activations_harvester.cache_mode != "no_cache":
        activations_cache_dir = os.path.join(cache_dir, "activations_cache")
        logger.info(f"Activations will be cached in: {activations_cache_dir}")

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
        cache_dir=activations_cache_dir,
        cache_mode=cfg.activations_harvester.cache_mode,
    )

    activations_dataloader = ScaledModelHookpointActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
    )

    return activations_dataloader
