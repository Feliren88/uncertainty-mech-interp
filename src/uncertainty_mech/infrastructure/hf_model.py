"""Hugging Face transformers adapter for the `LanguageModel` port.

Reads follow the ARENA 1.3.1 pattern: forward hooks on `model.model.layers[i]`
capture the residual stream at the last prompt token, and one more hook adds a
steering vector to a layer's output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uncertainty_mech.application.config import ModelConfig
from uncertainty_mech.application.ports import Readings, Steering
from uncertainty_mech.domain.prompts import ChatPrompt

log = logging.getLogger(__name__)


class HuggingFaceModel:
    def __init__(self, name: str, revision: str, dtype: str = "bfloat16", batch_size: int = 16, device: str = "cuda") -> None:
        self.name = f"{name}@{revision}"
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self._tokenizer.padding_side = "left"  # the last position is then the last prompt token for every row
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=getattr(torch, dtype), device_map=device
        ).eval()
        self._layers = self._model.model.layers
        self.num_layers = len(self._layers)

    @classmethod
    def from_config(cls, config: ModelConfig) -> HuggingFaceModel:
        return cls(config.name, config.revision, config.dtype, config.batch_size)

    def read(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        capture_residuals: bool = False,
        steering: Steering | None = None,
    ) -> Readings:
        letter_ids = [self._single_token(letter) for letter in letters]
        texts = [self._render(prompt) for prompt in prompts]
        lengths = [len(ids) for ids in self._tokenizer(texts, add_special_tokens=False).input_ids]
        order = np.argsort(lengths)[::-1]  # longest first, so the memory peak comes early
        logprobs = np.zeros((len(texts), len(letters)), dtype=np.float32)
        mass = np.zeros(len(texts), dtype=np.float32)
        hidden_size = self._model.config.hidden_size
        residuals = np.zeros((len(texts), self.num_layers, hidden_size), dtype=np.float32) if capture_residuals else None
        n_batches = (len(order) + self._batch_size - 1) // self._batch_size
        for number, start in enumerate(range(0, len(order), self._batch_size), start=1):
            batch = order[start : start + self._batch_size]
            batch_logprobs, batch_mass, batch_residuals = self._read_batch(
                [texts[i] for i in batch], letter_ids, capture_residuals, steering
            )
            logprobs[batch] = batch_logprobs
            mass[batch] = batch_mass
            if residuals is not None:
                residuals[batch] = batch_residuals
            if number % 50 == 0 or number == n_batches:
                log.info("Read batch %d/%d", number, n_batches)
        return Readings(logprobs, mass, residuals)

    def _single_token(self, letter: str) -> int:
        ids = self._tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {ids}")
        return ids[0]

    def _render(self, prompt: ChatPrompt) -> str:
        messages = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _read_batch(
        self, texts: list[str], letter_ids: list[int], capture: bool, steering: Steering | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self._model.device)
        # Left padding shifts the real tokens, so positions must count real tokens only.
        position_ids = (encoded.attention_mask.cumsum(-1) - 1).clamp(min=0)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        if steering is not None:  # registered first, so any capture at that layer sees the steered state
            handles.append(self._layers[steering.layer].register_forward_hook(self._steering_hook(steering)))
        if capture:
            handles += [layer.register_forward_hook(_capture_hook(index, captured)) for index, layer in enumerate(self._layers)]
        try:
            logits = self._model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :].float()
        finally:
            for handle in handles:
                handle.remove()
        letter_logprobs_full = torch.log_softmax(logits, dim=-1)[:, letter_ids]
        restricted = torch.log_softmax(logits[:, letter_ids], dim=-1)
        residuals = torch.stack([captured[i] for i in range(self.num_layers)], dim=1).numpy() if capture else None
        return restricted.cpu().numpy(), letter_logprobs_full.exp().sum(-1).cpu().numpy(), residuals

    def _steering_hook(self, steering: Steering) -> Callable:
        vector = torch.as_tensor(steering.direction * steering.scale, dtype=self._model.dtype, device=self._model.device)

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vector, *output[1:])
            return output + vector

        return hook


def _capture_hook(index: int, sink: dict[int, torch.Tensor]) -> Callable:
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sink[index] = hidden[:, -1, :].float().cpu()

    return hook
