from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from portable import _gelu_tanh, _layer_norm, _linear, _softmax
except ImportError:  # package import used by local tests
    from experiment7.reference.runtime_agent.portable import (
        _gelu_tanh,
        _layer_norm,
        _linear,
        _softmax,
    )


class PortableUniversalDeckTransformerPolicy:
    """NumPy inference for the Deck-8 autoregressive option/STOP policy."""

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

    def _deck_tokens(self, own_deck_cards: np.ndarray) -> np.ndarray:
        width = int(self.config["d_model"])
        heads = int(self.config["n_heads"])
        head_dim = width // heads
        cards = self._embedding("entity_embeddings.0.weight", own_deck_cards)
        queries = self.weights["deck_queries"]
        projection_weight = self.weights["deck_attention.in_proj_weight"]
        projection_bias = self.weights["deck_attention.in_proj_bias"]
        query = _linear(
            queries,
            projection_weight[:width],
            projection_bias[:width],
        )
        key = _linear(
            cards,
            projection_weight[width : 2 * width],
            projection_bias[width : 2 * width],
        )
        value = _linear(
            cards,
            projection_weight[2 * width :],
            projection_bias[2 * width :],
        )
        query = query.reshape(len(query), heads, head_dim).transpose(1, 0, 2)
        key = key.reshape(len(key), heads, head_dim).transpose(1, 0, 2)
        value = value.reshape(len(value), heads, head_dim).transpose(1, 0, 2)
        scores = np.matmul(query, key.transpose(0, 2, 1)) * np.float32(
            head_dim**-0.5
        )
        attention = _softmax(scores, axis=-1)
        attended = np.matmul(attention, value).transpose(1, 0, 2).reshape(-1, width)
        attended = _linear(
            attended,
            self.weights["deck_attention.out_proj.weight"],
            self.weights["deck_attention.out_proj.bias"],
        )
        eps = float(self.config["layer_norm_eps"])
        hidden = _layer_norm(
            queries + attended,
            self.weights["deck_norm1.weight"],
            self.weights["deck_norm1.bias"],
            eps,
        )
        feed_forward = _gelu_tanh(
            _linear(
                hidden,
                self.weights["deck_ff.0.weight"],
                self.weights["deck_ff.0.bias"],
            )
        )
        feed_forward = _linear(
            feed_forward,
            self.weights["deck_ff.2.weight"],
            self.weights["deck_ff.2.bias"],
        )
        return _layer_norm(
            hidden + feed_forward,
            self.weights["deck_norm2.weight"],
            self.weights["deck_norm2.bias"],
            eps,
        ).astype(np.float32, copy=False)

    def encode(
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
    ) -> dict[str, np.ndarray | float]:
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
        deck_size = int(self.config["deck_size"])
        if len(history_state) != history_length or len(history_action) != history_length:
            raise ValueError(f"expected {history_length} history slots")
        if len(own_deck_cards) != deck_size:
            raise ValueError(f"expected {deck_size} own deck cards")

        valid_entities = int(entity_mask.sum())
        entity_cat = entity_cat[:valid_entities]
        entity_num = entity_num[:valid_entities]
        entity_mask = entity_mask[:valid_entities]
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
        deck_token = self._deck_tokens(own_deck_cards) + token_type[2]
        entity_base = _linear(
            entity_num,
            self.weights["entity_num_projection.weight"],
            self.weights["entity_num_projection.bias"],
        )
        for field in range(entity_cat.shape[1] if entity_cat.ndim == 2 else 0):
            entity_base += self._embedding(
                f"entity_embeddings.{field}.weight", entity_cat[:, field]
            )
        opponent_mask = entity_cat[:, 2] == 1 if len(entity_cat) else np.zeros(0, dtype=bool)
        if bool(np.any(opponent_mask)):
            opponent_summary = entity_base[opponent_mask].mean(axis=0, keepdims=True)
        else:
            opponent_summary = np.zeros(
                (1, int(self.config["d_model"])), dtype=np.float32
            )
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
                deck_token,
                opponent_summary,
                entity_token,
                option_token,
            ),
            axis=0,
        ).astype(np.float32, copy=False)
        valid = np.concatenate(
            (
                np.ones(1, dtype=bool),
                history_mask,
                np.ones(int(self.config["deck_latents"]) + 1, dtype=bool),
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
            normalized = _layer_norm(
                x,
                self.weights[prefix + "ln1.weight"],
                self.weights[prefix + "ln1.bias"],
                eps,
            )
            qkv = _linear(
                normalized,
                self.weights[prefix + "qkv.weight"],
                self.weights[prefix + "qkv.bias"],
            ).reshape(len(x), 3, heads, head_dim)
            query = qkv[:, 0].transpose(1, 0, 2)
            key = qkv[:, 1].transpose(1, 0, 2)
            value = qkv[:, 2].transpose(1, 0, 2)
            scores = np.matmul(query, key.transpose(0, 2, 1)) * np.float32(
                head_dim**-0.5
            )
            scores[:, :, ~valid] = -1e4
            attention = _softmax(scores, axis=-1)
            mixed = np.matmul(attention, value).transpose(1, 0, 2).reshape(len(x), width)
            x = x + _linear(
                mixed,
                self.weights[prefix + "out.weight"],
                self.weights[prefix + "out.bias"],
            )
            normalized = _layer_norm(
                x,
                self.weights[prefix + "ln2.weight"],
                self.weights[prefix + "ln2.bias"],
                eps,
            )
            hidden = _gelu_tanh(
                _linear(
                    normalized,
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
        deck_begin = 1 + history_length
        opponent_index = deck_begin + int(self.config["deck_latents"])
        action_begin = opponent_index + 1 + valid_entities
        global_hidden = x[0]
        option_hidden = x[action_begin : action_begin + len(options)]
        value_logit = float(
            _linear(
                global_hidden.reshape(1, -1),
                self.weights["value_head.weight"],
                self.weights["value_head.bias"],
            )[0, 0]
        )
        opponent_logits = _linear(
            x[opponent_index : opponent_index + 1],
            self.weights["opponent_deck_head.weight"],
            self.weights["opponent_deck_head.bias"],
        ).reshape(-1)
        return {
            "global_hidden": global_hidden.astype(np.float32, copy=False),
            "option_hidden": option_hidden.astype(np.float32, copy=False),
            "option_mask": option_mask,
            "deck_hidden": x[deck_begin:opponent_index].astype(np.float32, copy=False),
            "value_logits": value_logit,
            "opponent_logits": opponent_logits.astype(np.float32, copy=False),
        }

    def decoder_logits(
        self,
        encoding: dict[str, np.ndarray | float],
        selected_mask: np.ndarray,
        min_count: int,
        max_count: int,
    ) -> np.ndarray:
        selected = np.asarray(selected_mask, dtype=bool).reshape(-1)
        option_hidden = np.asarray(encoding["option_hidden"], dtype=np.float32)
        option_mask = np.asarray(encoding["option_mask"], dtype=bool)
        selected_count = int(selected.sum())
        if selected_count:
            selected_mean = option_hidden[selected].mean(axis=0)
        else:
            selected_mean = np.zeros(int(self.config["d_model"]), dtype=np.float32)
        step = min(selected_count, int(self.config["max_actions"]))
        context_input = np.concatenate(
            (
                np.asarray(encoding["global_hidden"], dtype=np.float32),
                selected_mean,
                self.weights["decoder_step.weight"][step],
            )
        ).reshape(1, -1)
        context = np.tanh(
            _linear(
                context_input,
                self.weights["decoder_context.weight"],
                self.weights["decoder_context.bias"],
            )
        )[0]
        candidate_projection = option_hidden @ self.weights["decoder_option.weight"].T
        candidate = (
            (candidate_projection * context[None, :]).sum(axis=1)
            * np.float32(int(self.config["d_model"]) ** -0.5)
        )
        candidate += _linear(
            option_hidden,
            self.weights["decoder_option_bias.weight"],
            self.weights["decoder_option_bias.bias"],
        ).reshape(-1)
        candidate_valid = option_mask & ~selected & (selected_count < int(max_count))
        candidate = candidate.astype(np.float32, copy=False)
        candidate[~candidate_valid] = -1e4
        stop = float(
            _linear(
                context.reshape(1, -1),
                self.weights["decoder_stop.weight"],
                self.weights["decoder_stop.bias"],
            )[0, 0]
        )
        if selected_count < int(min_count):
            stop = -1e4
        return np.concatenate((candidate, np.asarray([stop], dtype=np.float32)))

    def greedy_actions(
        self,
        encoding: dict[str, np.ndarray | float],
        min_count: int,
        max_count: int,
    ) -> list[int]:
        options = len(np.asarray(encoding["option_mask"]))
        selected = np.zeros(options, dtype=bool)
        result: list[int] = []
        for _ in range(options + 1):
            choice = int(
                np.argmax(self.decoder_logits(encoding, selected, min_count, max_count))
            )
            if choice == options:
                break
            selected[choice] = True
            result.append(choice)
        return result

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
        encoding = self.encode(
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
        return (
            self.greedy_actions(encoding, min_count, max_count),
            np.asarray(encoding["opponent_logits"], dtype=np.float32),
        )


# The verified runtime main imports this historical symbol. Packaging replaces
# only the implementation module, keeping the public Agent interface unchanged.
PortableDeckIdentityTransformerPolicy = PortableUniversalDeckTransformerPolicy

