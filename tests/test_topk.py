
import torch

from crosscode.trainers.train_topk_crosscoder.trainer import aux_loss, topk_dead_latents
from crosscode.utils import l2_norm, not_none


def test_topk_dead_latents_with_k_less_than_dead():
    preacts_BL = torch.tensor([[5, 4, 3, 2, 1, 0]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 2

    expected_output_BL = torch.tensor([[0, 0, 0, 2, 1, 0]]).float()
    expected_n_used = 2  # k_aux, because it's less than the number of dead features

    aux_latents_BL, n_latents_used = not_none(
        topk_dead_latents(
            pre_activations_BL=preacts_BL,
            dead_features_mask_L=dead_features_mask_L,
            k_aux=k_aux,
        )
    )

    assert torch.allclose(aux_latents_BL, expected_output_BL)
    assert n_latents_used == expected_n_used


def test_topk_dead_latents_with_k_greater_than_dead():
    preacts_BL = torch.tensor([[5, 4, 3, 2, 1, 0]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 4

    expected_output_BL = torch.tensor([[0, 0, 0, 2, 1, 0]]).float()
    expected_n_used = 3  # because there are only 3 dead features (less than k_aux)

    aux_latents_BL, n_latents_used = not_none(
        topk_dead_latents(
            pre_activations_BL=preacts_BL,
            dead_features_mask_L=dead_features_mask_L,
            k_aux=k_aux,
        )
    )

    assert torch.allclose(aux_latents_BL, expected_output_BL)
    assert n_latents_used == expected_n_used


def test_aux_loss_with_k_aux_equal_to_dead():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 3
    decode_BD = lambda x: x  # noqa: E731
    error_BD = torch.tensor([[0, 0, 0, 4, 3, 2]]).float()

    expected_loss = 0.0  # should be able to reconstruct perfectly with the 3 dead features

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=decode_BD,
        error_BXD=error_BD,
    )

    assert loss.item() == expected_loss, f"loss should be {expected_loss}, but is {loss.item()}"


def test_aux_loss_with_k_aux_less_than_dead():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 2
    decode_BD = lambda x: x  # noqa: E731
    # Because of k < n_dead, the "2" should be unused, causing non-zero loss
    error_BD = torch.tensor([[0, 0, 0, 4, 3, 2]]).float()

    # we're using the l2 error mse, so the l2 error is 2 and therefore the loss is 4
    expected_loss = 4.0

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=decode_BD,
        error_BXD=error_BD,
    )

    assert loss.item() == expected_loss, f"loss should be {expected_loss}, but is {loss.item()}"


def test_aux_loss_with_k_aux_more_than_dead():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 4
    decode_BD = lambda x: x  # noqa: E731

    error_BD = torch.tensor([[0, 0, 0, 3, 2, 1]]).float()

    expected_aux_topk_BD = torch.tensor([[0, 0, 0, 4, 3, 2]]).float()

    expected_base_loss = l2_norm(error_BD - expected_aux_topk_BD) ** 2
    expected_loss_scale = 3 / 4  # because there are 3 dead features and k_aux is 4
    expected_loss = expected_base_loss * expected_loss_scale

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=decode_BD,
        error_BXD=error_BD,
    )

    assert loss.item() == expected_loss, f"loss should be {expected_loss}, but is {loss.item()}"


def test_aux_loss_with_no_dead_features():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 0, 0, 0]).bool()
    k_aux = 4

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=None,  # type: ignore
        error_BXD=None,  # type: ignore
    )

    assert loss.item() == 0.0
