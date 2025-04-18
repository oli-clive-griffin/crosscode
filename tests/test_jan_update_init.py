import torch

from crosscode.models.initialization.jan_update_init import compute_b_enc_L


def test_compute_b_enc_L():
    # d_model = 2  # as below in `batch_BD`
    n_latents = 2

    # define `get_pre_bias_BL` to be the identity function from d_model=2 -> n_latents=2 to make everything easier
    def get_pre_bias_BL(BD: torch.Tensor) -> torch.Tensor:
        return BD

    initial_approx_firing_pct = 0.25

    batch_BD = torch.tensor(
        [
            [0, 4],
            [1, 5],
            [2, 6],
            # this is where the threshold should be drawn. 25% of the examples should fire, i.e. just the example [3, 7]
            [3, 7],
        ]
    ).float()

    pre_biases = [get_pre_bias_BL(batch_BD), get_pre_bias_BL(batch_BD.clone()), get_pre_bias_BL(batch_BD.clone())]

    # use a jumprelu threshold of 2 for simplicity
    initial_jumprelu_threshold = 2
    initial_jumprelu_threshold_L = torch.ones(n_latents).float() * initial_jumprelu_threshold

    b_enc_L = compute_b_enc_L(
        pre_bias_iterator_BL=iter(pre_biases),
        initial_jumprelu_threshold_L=initial_jumprelu_threshold_L,
        initial_approx_firing_pct=initial_approx_firing_pct,
        n_tokens_for_threshold_setting=12,  # just the size of the dataset (3 batches of 2 examples each)
    )

    assert b_enc_L.shape == (n_latents,)

    # the threshold should be the 75th percentile of the pre-bias values, minus the jumprelu threshold
    # - 75th quantile of the pre-bias values is [2.25, 6.25] (see linear interpolation in torch.quantile)
    # - the jumprelu threshold is 2.0, so the threshold is 2 - [2.25, 6.25]
    expected_b_enc_L = torch.tensor([initial_jumprelu_threshold - 2.25, initial_jumprelu_threshold - 6.25])
    assert torch.allclose(b_enc_L, expected_b_enc_L), f"b_enc_L: {b_enc_L}, expected_b_enc_L: {expected_b_enc_L}"

    # Test that `initial_approx_firing_pct` of features fire when processing the batch

    pre_acts_BL = get_pre_bias_BL(batch_BD) + b_enc_L  # type: ignore
    acts_BL = (pre_acts_BL > initial_jumprelu_threshold) * pre_acts_BL
    assert (acts_BL != 0.0).float().mean() == initial_approx_firing_pct


def test_compute_b_enc_L_batches_rounding():
    n_latents = 1
    # d_model = 1  # as below in `batch_BD`

    # define `get_pre_bias_BL` to be the identity function from d_model=1 -> n_latents=1 to make everything easier
    def get_pre_bias_BL(BD: torch.Tensor) -> torch.Tensor:
        return BD

    initial_approx_firing_pct = 0.25

    batch_1_BD = torch.tensor([[0], [1], [2]]).float()
    batch_2_BD = torch.tensor([[3], [4], [5]]).float()

    pre_biases = [get_pre_bias_BL(batch_1_BD), get_pre_bias_BL(batch_2_BD)]

    initial_jumprelu_threshold_L = torch.randn(n_latents).float()

    b_enc_L = compute_b_enc_L(
        pre_bias_iterator_BL=iter(pre_biases),
        initial_jumprelu_threshold_L=initial_jumprelu_threshold_L,
        initial_approx_firing_pct=initial_approx_firing_pct,
        n_tokens_for_threshold_setting=5,  # should round up to 6 (taking in both batches)
    )

    assert b_enc_L.shape == (n_latents,)

    # the threshold should be the 75th percentile of the pre-bias values, minus the jumprelu threshold
    whole_dataset_BD = torch.cat([batch_1_BD, batch_2_BD])
    expected_b_enc_L = initial_jumprelu_threshold_L - torch.quantile(whole_dataset_BD, 1 - initial_approx_firing_pct)

    assert torch.allclose(b_enc_L, expected_b_enc_L), f"b_enc_L: {b_enc_L}, expected_b_enc_L: {expected_b_enc_L}"
