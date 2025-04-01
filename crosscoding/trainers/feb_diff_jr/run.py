import fire  # type: ignore

from crosscoding.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscoding.llms import build_llms
from crosscoding.log import logger
from crosscoding.models import (
    AnthropicSTEJumpReLUActivation,
    DataDependentJumpReLUInitStrategy,
    ModelHookpointAcausalCrosscoder,
)
from crosscoding.trainers.base_diffing_trainer import IdenticalLatentsInit
from crosscoding.trainers.base_trainer import run_exp
from crosscoding.trainers.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from crosscoding.trainers.feb_diff_jr.trainer import JumpReLUFebUpdateDiffingTrainer
from crosscoding.trainers.utils import build_wandb_run
from crosscoding.utils import get_device


def build_feb_update_crosscoder_trainer(
    cfg: JumpReLUModelDiffingFebUpdateExperimentConfig,
) -> JumpReLUFebUpdateDiffingTrainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 2, "expected 2 models for model-diffing"

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=[cfg.hookpoint],
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),
        hookpoints=[cfg.hookpoint],
        d_model=llms[0].cfg.d_model,
        n_latents=cfg.crosscoder.n_latents,
        init_strategy=IdenticalLatentsInit(
            first_init=DataDependentJumpReLUInitStrategy(
                activations_iterator_BMPD=(batch.activations_BMPD for batch in dataloader.get_activations_iterator()),
                initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
                n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
                device=device,
            ),
            n_shared_latents=cfg.crosscoder.n_shared_latents,
        ),
        activation_fn=AnthropicSTEJumpReLUActivation(
            size=cfg.crosscoder.n_latents,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
        ),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
    )

    return JumpReLUFebUpdateDiffingTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder.to(device),
        wandb_run=build_wandb_run(cfg),
        device=device,
        save_dir=cfg.save_dir,
        hookpoints=[cfg.hookpoint],
        n_shared_latents=cfg.crosscoder.n_shared_latents,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, JumpReLUModelDiffingFebUpdateExperimentConfig))
