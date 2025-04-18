import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import ReLUActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.models.initialization.anthropic_transpose import AnthropicTransposeInitCrossLayerTC
from crosscode.trainers.base_trainer import run_exp
from crosscode.trainers.l1_crosslayer_trancoder.config import L1CrossLayerTranscoderExperimentConfig
from crosscode.trainers.l1_crosslayer_trancoder.trainer import L1CrossLayerTranscoderTrainer
from crosscode.trainers.utils import build_wandb_run, get_activation_type
from crosscode.utils import get_device


def build_l1_cross_layer_transcoder_trainer(
    cfg: L1CrossLayerTranscoderExperimentConfig,
) -> L1CrossLayerTranscoderTrainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 1, "the trainer assumes we have one model"
    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=[cfg.in_hookpoint, *cfg.out_hookpoints],
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    activation_type = get_activation_type([cfg.in_hookpoint, *cfg.out_hookpoints])
    act_dim = llms[0].cfg.d_mlp if activation_type == "mlp" else llms[0].cfg.d_model

    crosscoder = CrossLayerTranscoder(
        d_model=act_dim,
        n_layers_out=len(cfg.out_hookpoints),
        n_latents=cfg.transcoder.n_latents,
        activation_fn=ReLUActivation(),
        use_encoder_bias=cfg.transcoder.use_encoder_bias,
        use_decoder_bias=cfg.transcoder.use_decoder_bias,
        init_strategy=AnthropicTransposeInitCrossLayerTC(dec_init_norm=cfg.transcoder.dec_init_norm),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return L1CrossLayerTranscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        model=crosscoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir,
        out_hookpoints=cfg.out_hookpoints,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_l1_cross_layer_transcoder_trainer, L1CrossLayerTranscoderExperimentConfig))
