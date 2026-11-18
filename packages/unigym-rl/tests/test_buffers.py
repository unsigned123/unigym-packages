"""R2.5 contract tests for :class:`unigym_rl.buffers.RingBuffer`."""

from __future__ import annotations

import pytest
import torch

from unigym_rl import RingBuffer


def test_latest_and_history_are_chronological_over_wraparound() -> None:
    buffer = RingBuffer(num_envs=1, capacity=3, shape=(1,))
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        buffer.push(torch.tensor([[value]]))

    assert buffer.count.tolist() == [3]
    assert buffer.latest(1).squeeze(-1).tolist() == [[5.0]]
    assert buffer.latest(2).squeeze(-1).tolist() == [[4.0, 5.0]]
    assert buffer.latest(3).squeeze(-1).tolist() == [[3.0, 4.0, 5.0]]
    assert buffer.history().squeeze(-1).tolist() == [[3.0, 4.0, 5.0]]


def test_per_env_pointers_are_independent_and_reset_rows_stay_cleared() -> None:
    buffer = RingBuffer(num_envs=2, capacity=2)
    buffer.push(torch.tensor([10.0, 20.0]))
    buffer.push(torch.tensor([11.0, 21.0]))
    assert buffer.history().tolist() == [[10.0, 11.0], [20.0, 21.0]]

    buffer.reset([1])
    buffer.push(torch.tensor([12.0, 22.0]))

    assert buffer.count.tolist() == [2, 1]
    assert buffer.history().tolist() == [[11.0, 12.0], [0.0, 22.0]]


def test_zero_fill_before_full() -> None:
    buffer = RingBuffer(num_envs=2, capacity=3, shape=(2,))
    buffer.push(torch.ones(2, 2))

    latest = buffer.latest(3)

    assert latest.shape == (2, 3, 2)
    assert torch.equal(latest[:, :2], torch.zeros(2, 2, 2))
    assert torch.equal(latest[:, 2], torch.ones(2, 2))
    assert buffer.count.tolist() == [1, 1]


def test_full_reset_clears_data_pointers_and_counts() -> None:
    buffer = RingBuffer(num_envs=2, capacity=2, shape=(1,))
    buffer.push(torch.ones(2, 1))
    buffer.push(torch.full((2, 1), 2.0))

    buffer.reset()

    assert buffer.count.tolist() == [0, 0]
    assert torch.equal(buffer.data, torch.zeros(2, 2, 1))
    assert torch.equal(buffer.history(), torch.zeros(2, 2, 1))

    buffer.push(torch.full((2, 1), 7.0))
    assert buffer.latest(1).squeeze(-1).tolist() == [[7.0], [7.0]]


def test_dtype_and_device_are_normalized() -> None:
    buffer = RingBuffer(num_envs=1, capacity=2, dtype=torch.float64, device="cpu")
    buffer.push(torch.tensor([1], dtype=torch.int32))

    assert buffer.data.dtype is torch.float64
    assert buffer.data.device == torch.device("cpu")
    assert buffer.latest(1).dtype is torch.float64
    assert buffer.latest(1).item() == pytest.approx(1.0)


@pytest.mark.parametrize(("num_envs", "capacity"), [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_invalid_sizes_are_rejected(num_envs: int, capacity: int) -> None:
    with pytest.raises(ValueError):
        RingBuffer(num_envs=num_envs, capacity=capacity)


def test_non_integer_sizes_are_rejected() -> None:
    with pytest.raises(TypeError):
        RingBuffer(num_envs="1", capacity=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RingBuffer(num_envs=1, capacity=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RingBuffer(num_envs=1, capacity=1, shape=[2])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RingBuffer(num_envs=1, capacity=1, dtype="float32")  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [0, -1, 4])
def test_invalid_latest_n_is_rejected(n: int) -> None:
    buffer = RingBuffer(num_envs=1, capacity=3)
    with pytest.raises(ValueError):
        buffer.latest(n)


def test_non_integer_latest_n_is_rejected() -> None:
    buffer = RingBuffer(num_envs=1, capacity=3)
    with pytest.raises(TypeError):
        buffer.latest(1.0)  # type: ignore[arg-type]


def test_bad_push_shape_and_type_are_rejected() -> None:
    buffer = RingBuffer(num_envs=2, capacity=2, shape=(1,))
    with pytest.raises(ValueError):
        buffer.push(torch.zeros(2))
    with pytest.raises(ValueError):
        buffer.push(torch.zeros(2, 2, 1))
    with pytest.raises(TypeError):
        buffer.push([[0.0], [0.0]])  # type: ignore[arg-type]


def test_bad_reset_ids_are_rejected() -> None:
    buffer = RingBuffer(num_envs=2, capacity=2)
    with pytest.raises(ValueError):
        buffer.reset(torch.tensor([2]))
    with pytest.raises(ValueError):
        buffer.reset(torch.tensor([-1]))
    with pytest.raises(TypeError):
        buffer.reset(torch.tensor([True, False]))
    with pytest.raises(TypeError):
        buffer.reset("0")
    with pytest.raises(TypeError):
        buffer.reset(0)  # type: ignore[arg-type]
