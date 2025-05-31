from typing import cast

import fire  # type: ignore
import torch
from transformer_lens.components.mlps.mlp import MLP
from transformer_lens.components.transformer_block import TransformerBlock

from crosscode.data.activation_harvester import get_layer
from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models.acausal_crosscoder import RichReLUTranscoder, RichTranscoderWrapper
from crosscode.trainers.topk_crosscoder.config import TopKAcausalCrosscoderExperimentConfig
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_trainer(cfg: TopKAcausalCrosscoderExperimentConfig) -> Trainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    # match cfg.train.topk_style:
    #     case "topk":
    #         cc_act = TopkActivation(k=cfg.crosscoder.k)
    #     case "batch_topk":
    #         cc_act = BatchTopkActivation(k_per_example=cfg.crosscoder.k)
    #     case "groupmax":
    #         cc_act = GroupMaxActivation(k_groups=cfg.crosscoder.k, latents_size=cfg.crosscoder.n_latents)

    
    d_model = llms[0].cfg.d_model

    d_hidden = llms[0].cfg.d_mlp

    # tc = RichSwiGLUTranscoderWrapper(
    #     model=RichSwiGLUTranscoder(


    assert len(cfg.hookpoints) == 2
    layer_idx = get_layer(cfg.hookpoints[0])
    block = cast(TransformerBlock, llms[0].blocks[layer_idx])
    layer = cast(MLP, block.mlp)

    tc = RichTranscoderWrapper(
        model=RichReLUTranscoder(
            d_model=d_model,
            d_hidden=d_hidden,
            n_latents=cfg.crosscoder.n_latents,
            k=cfg.crosscoder.k,
            # latent_activation_fn=cc_act,
            ref_W_in=layer.W_in,
        )
    )


    # with torch.no_grad():
    #     tc.model.mlp_W_up_DH.copy_(layer.W_in)
    #     tc.model.sparse_dec_LD.normal_()
    #     tc.model.sparse_enc_HL.normal_()

    llms_cfg = cfg.data.activations_harvester.llms
    assert len(llms_cfg) == 1
    llm_cfg = llms_cfg[0]
    assert llm_cfg.name is not None
    # assert llm_cfg.name.startswith("google/gemma-2")

    # if cfg.train.k_aux is None:
    #     cfg.train.k_aux = d_model // 2
    #     logger.info(f"defaulting to k_aux={cfg.train.k_aux} for crosscoder (({d_model=}) // 2)")

    wandb_run = build_wandb_run(cfg)

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    return Trainer(
        activations_dataloader=dataloader,
        model=tc,
        optimizer_cfg=cfg.train.optimizer,
        wandb_run=wandb_run,
        # make this into a "train loop cfg"?
        num_steps=cfg.train.num_steps,
        gradient_accumulation_microbatches_per_step=cfg.train.gradient_accumulation_microbatches_per_step,
        save_every_n_steps=cfg.train.save_every_n_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        upload_saves_to_wandb=cfg.train.upload_saves_to_wandb,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKAcausalCrosscoderExperimentConfig))
