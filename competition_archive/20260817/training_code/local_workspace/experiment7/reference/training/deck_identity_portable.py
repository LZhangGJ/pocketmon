from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from portable import _gelu_tanh, _layer_norm, _linear, _softmax


class PortableDeckIdentityTransformerPolicy:
    def __init__(self, model_path: str | Path) -> None:
        with np.load(model_path) as archive:
            self.weights = {
                name: archive[name].astype(np.float32, copy=False)
                for name in archive.files
                if name != "config_json"
            }
            self.config: dict[str, Any] = json.loads(str(archive["config_json"][0]))

    def _embedding(self, name: str, indices: np.ndarray) -> np.ndarray:
        return self.weights[name][indices]

    def forward(
        self,
        state: np.ndarray,
        history_state: np.ndarray,
        history_action: np.ndarray,
        history_mask: np.ndarray,
        own_deck_cards: np.ndarray,
        entity_cat: np.ndarray,
        entity_num: np.ndarray,
        entity_mask: np.ndarray,
        options: np.ndarray,
        option_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        history_state = np.asarray(history_state, dtype=np.float32)
        history_action = np.asarray(history_action, dtype=np.float32)
        history_mask = np.asarray(history_mask, dtype=bool).reshape(-1)
        own_deck_cards = np.asarray(own_deck_cards, dtype=np.int64).reshape(-1)
        entity_cat = np.asarray(entity_cat, dtype=np.int64)
        entity_num = np.asarray(entity_num, dtype=np.float32)
        entity_mask = np.asarray(entity_mask, dtype=bool).reshape(-1)
        options = np.asarray(options, dtype=np.float32)
        if option_mask is None:
            option_mask = np.ones(len(options), dtype=bool)
        else:
            option_mask = np.asarray(option_mask, dtype=bool).reshape(-1)
        history_length = int(self.config["history_length"])
        if len(history_state) != history_length or len(history_action) != history_length:
            raise ValueError(f"expected {history_length} history slots")
        if len(own_deck_cards) != int(self.config["deck_size"]):
            raise ValueError(f"expected {self.config['deck_size']} own deck cards")

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
        history_token = (
            _linear(
                history_state,
                self.weights["history_state_projection.weight"],
                self.weights["history_state_projection.bias"],
            )
            + _linear(
                history_action,
                self.weights["history_action_projection.weight"],
                self.weights["history_action_projection.bias"],
            )
            + self.weights["history_position.weight"]
            + token_type[1]
        )
        own_deck_token = self._embedding(
            "entity_embeddings.0.weight", own_deck_cards
        ).mean(axis=0, keepdims=True) + token_type[2]
        entity_base = _linear(
            entity_num,
            self.weights["entity_num_projection.weight"],
            self.weights["entity_num_projection.bias"],
        )
        for field in range(entity_cat.shape[1]):
            entity_base += self._embedding(
                f"entity_embeddings.{field}.weight", entity_cat[:, field]
            )
        opponent_mask = entity_cat[:, 2] == 1
        if np.any(opponent_mask):
            opponent_summary = entity_base[opponent_mask].mean(axis=0, keepdims=True)
        else:
            opponent_summary = np.zeros((1, int(self.config["d_model"])), dtype=np.float32)
        opponent_summary += token_type[3]
        entity_token = entity_base + token_type[4]
        option_token = _linear(
            options,
            self.weights["option_projection.weight"],
            self.weights["option_projection.bias"],
        ) + token_type[5]
        x = np.concatenate(
            (
                global_token,
                history_token,
                own_deck_token,
                opponent_summary,
                entity_token,
                option_token,
            ),
            axis=0,
        ).astype(np.float32)
        valid = np.concatenate(
            (
                np.ones(1, dtype=bool),
                history_mask,
                np.ones(2, dtype=bool),
                entity_mask,
                option_mask,
            )
        )

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
            qkv = _linear(
                norm,
                self.weights[prefix + "qkv.weight"],
                self.weights[prefix + "qkv.bias"],
            ).reshape(len(x), 3, heads, head_dim)
            q = qkv[:, 0].transpose(1, 0, 2)
            k = qkv[:, 1].transpose(1, 0, 2)
            v = qkv[:, 2].transpose(1, 0, 2)
            scores = np.matmul(q, k.transpose(0, 2, 1)) * (head_dim ** -0.5)
            scores[:, :, ~valid] = -1e4
            attention = _softmax(scores, axis=-1)
            mixed = np.matmul(attention, v).transpose(1, 0, 2).reshape(len(x), width)
            x = x + _linear(
                mixed,
                self.weights[prefix + "out.weight"],
                self.weights[prefix + "out.bias"],
            )
            norm = _layer_norm(
                x,
                self.weights[prefix + "ln2.weight"],
                self.weights[prefix + "ln2.bias"],
                eps,
            )
            hidden = _gelu_tanh(
                _linear(
                    norm,
                    self.weights[prefix + "fc1.weight"],
                    self.weights[prefix + "fc1.bias"],
                )
            )
            x = x + _linear(
                hidden,
                self.weights[prefix + "fc2.weight"],
                self.weights[prefix + "fc2.bias"],
            )

        x = _layer_norm(
            x,
            self.weights["final_norm.weight"],
            self.weights["final_norm.bias"],
            eps,
        )
        opponent_index = 1 + history_length + 1
        action_begin = opponent_index + 1 + valid_entities
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
            _linear(
                x[0:1],
                self.weights["value_head.weight"],
                self.weights["value_head.bias"],
            )[0, 0]
        )
        opponent_logits = _linear(
            x[opponent_index : opponent_index + 1],
            self.weights["opponent_deck_head.weight"],
            self.weights["opponent_deck_head.bias"],
        ).reshape(-1)
        return (
            option_logits.astype(np.float32),
            count_logits.astype(np.float32),
            value_logit,
            opponent_logits.astype(np.float32),
        )

    def choose(
        self,
        state: np.ndarray,
        history_state: np.ndarray,
        history_action: np.ndarray,
        history_mask: np.ndarray,
        own_deck_cards: np.ndarray,
        entity_cat: np.ndarray,
        entity_num: np.ndarray,
        entity_mask: np.ndarray,
        options: np.ndarray,
        min_count: int,
        max_count: int,
    ) -> tuple[list[int], np.ndarray]:
        option_logits, count_logits, _, opponent_logits = self.forward(
            state,
            history_state,
            history_action,
            history_mask,
            own_deck_cards,
            entity_cat,
            entity_num,
            entity_mask,
            options,
        )
        count_limit = min(max_count, len(option_logits), len(count_logits) - 1)
        min_limit = min(min_count, count_limit)
        if min_limit == count_limit:
            count = count_limit
        else:
            masked = np.full_like(count_logits, -1e4)
            masked[min_limit : count_limit + 1] = count_logits[
                min_limit : count_limit + 1
            ]
            count = int(np.argmax(masked))
        order = np.argsort(-option_logits, kind="stable")
        return [int(value) for value in order[:count]], opponent_logits
