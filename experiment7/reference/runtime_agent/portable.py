from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=axis, keepdims=True), 1e-12)


class PortableTransformerPolicy:
    def __init__(self, model_path: str | Path) -> None:
        with np.load(model_path) as archive:
            self.weights = {name: archive[name].astype(np.float32, copy=False) for name in archive.files if name != "config_json"}
            self.config: dict[str, Any] = json.loads(str(archive["config_json"][0]))

    def _embedding(self, name: str, indices: np.ndarray) -> np.ndarray:
        return self.weights[name][indices]

    def forward(
        self,
        state: np.ndarray,
        entity_cat: np.ndarray,
        entity_num: np.ndarray,
        entity_mask: np.ndarray,
        options: np.ndarray,
        option_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        entity_cat = np.asarray(entity_cat, dtype=np.int64)
        entity_num = np.asarray(entity_num, dtype=np.float32)
        entity_mask = np.asarray(entity_mask, dtype=bool).reshape(-1)
        options = np.asarray(options, dtype=np.float32)
        if option_mask is None:
            option_mask = np.ones(len(options), dtype=bool)
        else:
            option_mask = np.asarray(option_mask, dtype=bool).reshape(-1)

        valid_entities = int(entity_mask.sum())
        entity_cat = entity_cat[:valid_entities]
        entity_num = entity_num[:valid_entities]
        entity_mask = entity_mask[:valid_entities]
        option_count = len(options)

        token_type = self.weights["token_type.weight"]
        global_token = _linear(
            state,
            self.weights["state_projection.weight"],
            self.weights["state_projection.bias"],
        ) + token_type[0]
        entity_token = _linear(
            entity_num,
            self.weights["entity_num_projection.weight"],
            self.weights["entity_num_projection.bias"],
        )
        for field in range(entity_cat.shape[1]):
            entity_token += self._embedding(f"entity_embeddings.{field}.weight", entity_cat[:, field])
        entity_token += token_type[1]
        option_token = _linear(
            options,
            self.weights["option_projection.weight"],
            self.weights["option_projection.bias"],
        ) + token_type[2]
        x = np.concatenate((global_token, entity_token, option_token), axis=0).astype(np.float32)
        valid = np.concatenate((np.ones(1, dtype=bool), entity_mask, option_mask))

        heads = int(self.config["n_heads"])
        width = int(self.config["d_model"])
        head_dim = width // heads
        eps = float(self.config["layer_norm_eps"])
        for layer in range(int(self.config["n_layers"])):
            prefix = f"blocks.{layer}."
            norm = _layer_norm(
                x,
                self.weights[prefix + "ln1.weight"],
                self.weights[prefix + "ln1.bias"],
                eps,
            )
            qkv = _linear(norm, self.weights[prefix + "qkv.weight"], self.weights[prefix + "qkv.bias"])
            qkv = qkv.reshape(len(x), 3, heads, head_dim)
            q = qkv[:, 0].transpose(1, 0, 2)
            k = qkv[:, 1].transpose(1, 0, 2)
            v = qkv[:, 2].transpose(1, 0, 2)
            scores = np.matmul(q, k.transpose(0, 2, 1)) * (head_dim ** -0.5)
            scores[:, :, ~valid] = -1e4
            attention = _softmax(scores, axis=-1)
            mixed = np.matmul(attention, v).transpose(1, 0, 2).reshape(len(x), width)
            x = x + _linear(mixed, self.weights[prefix + "out.weight"], self.weights[prefix + "out.bias"])
            norm = _layer_norm(
                x,
                self.weights[prefix + "ln2.weight"],
                self.weights[prefix + "ln2.bias"],
                eps,
            )
            hidden = _gelu_tanh(
                _linear(norm, self.weights[prefix + "fc1.weight"], self.weights[prefix + "fc1.bias"])
            )
            x = x + _linear(hidden, self.weights[prefix + "fc2.weight"], self.weights[prefix + "fc2.bias"])

        x = _layer_norm(
            x,
            self.weights["final_norm.weight"],
            self.weights["final_norm.bias"],
            eps,
        )
        action_begin = 1 + valid_entities
        action_hidden = x[action_begin : action_begin + option_count]
        option_logits = _linear(
            action_hidden,
            self.weights["option_head.weight"],
            self.weights["option_head.bias"],
        ).reshape(-1)
        option_logits[~option_mask] = -1e4
        count_logits = _linear(
            x[0:1], self.weights["count_head.weight"], self.weights["count_head.bias"]
        ).reshape(-1)
        value_logit = float(
            _linear(x[0:1], self.weights["value_head.weight"], self.weights["value_head.bias"])[0, 0]
        )
        return option_logits.astype(np.float32), count_logits.astype(np.float32), value_logit

    def choose(
        self,
        state: np.ndarray,
        entity_cat: np.ndarray,
        entity_num: np.ndarray,
        entity_mask: np.ndarray,
        options: np.ndarray,
        min_count: int,
        max_count: int,
    ) -> list[int]:
        option_logits, count_logits, _ = self.forward(
            state, entity_cat, entity_num, entity_mask, options
        )
        count_limit = min(max_count, len(option_logits), len(count_logits) - 1)
        min_limit = min(min_count, count_limit)
        if min_limit == count_limit:
            count = count_limit
        else:
            masked = np.full_like(count_logits, -1e4)
            masked[min_limit : count_limit + 1] = count_logits[min_limit : count_limit + 1]
            count = int(np.argmax(masked))
        order = np.argsort(-option_logits, kind="stable")
        return [int(value) for value in order[:count]]

