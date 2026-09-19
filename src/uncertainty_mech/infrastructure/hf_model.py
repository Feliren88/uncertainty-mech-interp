"""Hugging Face transformers adapter for the `LanguageModel` and `CircuitModel` ports.

It follows the ARENA 1.3.1 pattern: forward hooks read and edit activations.
Residual stream: output of `model.model.layers[i]`. Attention heads: the input
of `self_attn.o_proj`, where head outputs sit side by side before mixing. MLP:
output of `layers[i].mlp`. Each hooked module gets one hook that applies its
patches, then its additions, then its captures.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uncertainty_mech.application.config import ModelConfig
from uncertainty_mech.application.ports import (
    Addition,
    Capture,
    Patch,
    Readings,
    Site,
    SiteKind,
    Steering,
    Trace,
)
from uncertainty_mech.domain.prompts import ChatPrompt

log = logging.getLogger(__name__)


@dataclass
class _ModuleEdits:
    patches: list[Patch] = field(default_factory=list)
    additions: list[Addition] = field(default_factory=list)
    captures: list[tuple[int, Capture]] = field(default_factory=list)


class HuggingFaceModel:
    def __init__(
        self, name: str, revision: str, dtype: str = "bfloat16", batch_size: int = 16, device: str = "cuda"
    ) -> None:
        self.name = f"{name}@{revision}"
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self._tokenizer.padding_side = "left"  # position -k is then the k-th last prompt token in every row
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=getattr(torch, dtype), device_map=device
        ).eval()
        self._layers = self._model.model.layers
        config = self._model.config
        self.num_layers = len(self._layers)
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

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
        captures = (
            [Capture(Site(SiteKind.RESID, layer), (1,)) for layer in range(self.num_layers)]
            if capture_residuals
            else []
        )
        additions = (
            []
            if steering is None
            else [
                Addition(
                    Site(SiteKind.RESID, steering.layer),
                    None,
                    steering.direction,
                    np.full(len(prompts), steering.scale, dtype=np.float32),
                )
            ]
        )
        trace = self.trace(prompts, letters, captures=captures, additions=additions)
        residuals = np.stack([values[:, 0] for values in trace.activations], axis=1) if captures else None
        return Readings(trace.letter_logprobs, trace.letter_mass, residuals)

    def common_suffix_length(self, prompts: Sequence[ChatPrompt]) -> int:
        token_lists = [self._tokenizer(self._render(p), add_special_tokens=False).input_ids for p in prompts]
        shortest = min(len(tokens) for tokens in token_lists)
        length = 0
        while length < shortest and all(tokens[-1 - length] == token_lists[0][-1 - length] for tokens in token_lists):
            length += 1
        return length

    def trace(
        self,
        prompts: Sequence[ChatPrompt],
        letters: Sequence[str],
        *,
        captures: Sequence[Capture] = (),
        patches: Sequence[Patch] = (),
        additions: Sequence[Addition] = (),
    ) -> Trace:
        letter_ids = [self._single_token(letter) for letter in letters]
        texts = [self._render(prompt) for prompt in prompts]
        lengths = [len(ids) for ids in self._tokenizer(texts, add_special_tokens=False).input_ids]
        order = np.argsort(lengths)[::-1]  # longest first, so the memory peak comes early
        logprobs = np.zeros((len(texts), len(letters)), dtype=np.float32)
        mass = np.zeros(len(texts), dtype=np.float32)
        activations = [
            np.zeros((len(texts), _offset_count(c.offsets), self._width(c.site)), dtype=np.float32) for c in captures
        ]
        edits = _group_by_module(captures, patches, additions)
        n_batches = (len(order) + self._batch_size - 1) // self._batch_size
        for number, start in enumerate(range(0, len(order), self._batch_size), start=1):
            rows = order[start : start + self._batch_size]
            batch_logprobs, batch_mass, batch_activations = self._trace_batch(
                [texts[i] for i in rows], rows, letter_ids, edits
            )
            logprobs[rows] = batch_logprobs
            mass[rows] = batch_mass
            for index, values in batch_activations.items():
                activations[index][rows] = values
            if number % 50 == 0 or number == n_batches:
                log.info("Read batch %d/%d", number, n_batches)
        return Trace(logprobs, mass, activations)

    def _width(self, site: Site) -> int:
        if site.kind is SiteKind.ATTN_Z:
            return self.head_dim if site.head is not None else self.num_heads * self.head_dim
        return self._model.config.hidden_size

    def _single_token(self, letter: str) -> int:
        ids = self._tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {ids}")
        return ids[0]

    def _render(self, prompt: ChatPrompt) -> str:
        messages = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _trace_batch(
        self, texts: list[str], rows: np.ndarray, letter_ids: list[int], edits: dict[tuple[SiteKind, int], _ModuleEdits]
    ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self._model.device
        )
        lengths = encoded.attention_mask.sum(-1).cpu().numpy()
        # Left padding shifts the real tokens, so positions must count real tokens only.
        position_ids = (encoded.attention_mask.cumsum(-1) - 1).clamp(min=0)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for (kind, layer), module_edits in edits.items():
            editor = _Editor(module_edits, rows, lengths, self.head_dim, captured)
            if kind is SiteKind.RESID:
                handles.append(self._layers[layer].register_forward_hook(editor.output_hook))
            elif kind is SiteKind.MLP:
                handles.append(self._layers[layer].mlp.register_forward_hook(editor.output_hook))
            else:
                handles.append(self._layers[layer].self_attn.o_proj.register_forward_pre_hook(editor.input_hook))
        try:
            logits = (
                self._model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[:, -1, :]
                .float()
            )
        finally:
            for handle in handles:
                handle.remove()
        letter_logprobs_full = torch.log_softmax(logits, dim=-1)[:, letter_ids]
        restricted = torch.log_softmax(logits[:, letter_ids], dim=-1)
        return (
            restricted.cpu().numpy(),
            letter_logprobs_full.exp().sum(-1).cpu().numpy(),
            {index: values.numpy() for index, values in captured.items()},
        )


def _offset_count(offsets) -> int:
    return offsets.shape[1] if isinstance(offsets, np.ndarray) else len(offsets)


def _group_by_module(
    captures: Sequence[Capture], patches: Sequence[Patch], additions: Sequence[Addition]
) -> dict[tuple[SiteKind, int], _ModuleEdits]:
    edits: dict[tuple[SiteKind, int], _ModuleEdits] = {}
    for patch in patches:
        edits.setdefault((patch.site.kind, patch.site.layer), _ModuleEdits()).patches.append(patch)
    for addition in additions:
        edits.setdefault((addition.site.kind, addition.site.layer), _ModuleEdits()).additions.append(addition)
    for index, capture in enumerate(captures):
        edits.setdefault((capture.site.kind, capture.site.layer), _ModuleEdits()).captures.append((index, capture))
    return edits


class _Editor:
    """Applies one module's patches, additions and captures to a [batch, seq, width] tensor."""

    def __init__(
        self,
        edits: _ModuleEdits,
        rows: np.ndarray,
        lengths: np.ndarray,
        head_dim: int,
        sink: dict[int, torch.Tensor],
    ) -> None:
        self._edits = edits
        self._rows = rows
        self._lengths = lengths  # real tokens per row in this batch
        self._head_dim = head_dim
        self._sink = sink

    def output_hook(self, module, inputs, output):
        if isinstance(output, tuple):
            return (self._apply(output[0]), *output[1:])
        return self._apply(output)

    def input_hook(self, module, args):
        return (self._apply(args[0]), *args[1:])

    def _apply(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._edits.patches or self._edits.additions:
            hidden = hidden.clone()
        for patch in self._edits.patches:
            batch, positions = self._index(hidden, patch.offsets)
            values = torch.as_tensor(patch.values[self._rows], dtype=hidden.dtype, device=hidden.device)
            hidden[batch, positions, self._columns(patch.site)] = values
        for addition in self._edits.additions:
            vector = torch.as_tensor(addition.vector, dtype=hidden.dtype, device=hidden.device)
            scale = torch.as_tensor(addition.scale[self._rows], dtype=hidden.dtype, device=hidden.device)[:, None, None]
            columns = self._columns(addition.site)
            if addition.offsets is None:
                hidden[:, :, columns] += scale * vector
            else:
                batch, positions = self._index(hidden, addition.offsets)
                hidden[batch, positions, columns] += scale * vector
        for index, capture in self._edits.captures:
            batch, positions = self._index(hidden, capture.offsets)
            self._sink[index] = hidden[batch, positions, self._columns(capture.site)].float().cpu()
        return hidden

    def _index(self, hidden: torch.Tensor, offsets) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch and position indices [b, k] for offsets counted from the end of each row."""
        per_row = offsets[self._rows] if isinstance(offsets, np.ndarray) else np.tile(offsets, (len(self._rows), 1))
        if (per_row.max(axis=1) > self._lengths).any() or (per_row < 1).any():
            raise ValueError("an offset reaches outside a prompt")
        positions = torch.as_tensor(hidden.shape[1] - per_row, device=hidden.device)
        batch = torch.arange(len(self._rows), device=hidden.device)[:, None].expand_as(positions)
        return batch, positions

    def _columns(self, site: Site) -> slice:
        if site.head is None:
            return slice(None)
        return slice(site.head * self._head_dim, (site.head + 1) * self._head_dim)
