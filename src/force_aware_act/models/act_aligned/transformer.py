"""Batch-first Transformer blocks preserving official ACT/DETR semantics."""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import ACTAlignedConfig
from force_aware_act.models.act_aligned.contracts import require_padding_mask


class ACTEncoderLayer(nn.Module):
    """Post-norm encoder layer with positions added only to attention q/k."""

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        source: torch.Tensor,
        *,
        position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _require_tokens(source, "source", self.d_model)
        _require_optional_position(position, source, "position")
        if padding_mask is not None:
            require_padding_mask(
                padding_mask,
                name="padding_mask",
                batch_size=source.shape[0],
                sequence_length=source.shape[1],
            )

        query_key = source if position is None else source + position
        attended = self.self_attn(
            query_key,
            query_key,
            source,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        source = self.norm1(source + self.dropout1(attended))
        feedforward = self.linear2(
            self.dropout(self.activation(self.linear1(source)))
        )
        return self.norm2(source + self.dropout2(feedforward))


class ACTTransformerEncoder(nn.Module):
    """Official-depth ACT encoder without a post-stack LayerNorm."""

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        layer = ACTEncoderLayer(config)
        self.layers = nn.ModuleList(
            copy.deepcopy(layer) for _ in range(config.encoder_layers)
        )
        self.num_layers = config.encoder_layers
        self.d_model = config.d_model
        _reset_xavier(self)

    def forward(
        self,
        source: torch.Tensor,
        *,
        position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = source
        for layer in self.layers:
            output = layer(
                output,
                position=position,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )
        return output


class ACTDecoderLayer(nn.Module):
    """Post-norm ACT decoder layer with query and memory positions."""

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_position: Optional[torch.Tensor] = None,
        memory_position: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _require_tokens(target, "target", self.d_model)
        _require_tokens(memory, "memory", self.d_model)
        if target.shape[0] != memory.shape[0]:
            raise ValueError("target and memory batch sizes must match")
        _require_optional_position(query_position, target, "query_position")
        _require_optional_position(memory_position, memory, "memory_position")
        if target_padding_mask is not None:
            require_padding_mask(
                target_padding_mask,
                name="target_padding_mask",
                batch_size=target.shape[0],
                sequence_length=target.shape[1],
            )
        if memory_padding_mask is not None:
            require_padding_mask(
                memory_padding_mask,
                name="memory_padding_mask",
                batch_size=memory.shape[0],
                sequence_length=memory.shape[1],
            )

        target_query_key = (
            target if query_position is None else target + query_position
        )
        attended = self.self_attn(
            target_query_key,
            target_query_key,
            target,
            attn_mask=target_attention_mask,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )[0]
        target = self.norm1(target + self.dropout1(attended))

        cross_query = target if query_position is None else target + query_position
        cross_key = memory if memory_position is None else memory + memory_position
        attended = self.cross_attn(
            cross_query,
            cross_key,
            memory,
            attn_mask=memory_attention_mask,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        target = self.norm2(target + self.dropout2(attended))

        feedforward = self.linear2(
            self.dropout(self.activation(self.linear1(target)))
        )
        return self.norm3(target + self.dropout3(feedforward))


class ACTTransformerDecoder(nn.Module):
    """ACT decoder stack with a shared final LayerNorm."""

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        layer = ACTDecoderLayer(config)
        self.layers = nn.ModuleList(
            copy.deepcopy(layer) for _ in range(config.decoder_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.num_layers = config.decoder_layers
        self.d_model = config.d_model
        _reset_xavier(self)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_position: Optional[torch.Tensor] = None,
        memory_position: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        output = target
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                query_position=query_position,
                memory_position=memory_position,
                target_attention_mask=target_attention_mask,
                memory_attention_mask=memory_attention_mask,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )
            if return_intermediate:
                intermediate.append(self.norm(output))

        output = self.norm(output)
        if return_intermediate:
            intermediate[-1] = output
            return torch.stack(intermediate)
        return output


class ACTQueryDecoder(nn.Module):
    """Decode one learned query per action-chunk step.

    Input memory: ``[B, S_memory, d_model]``
    Final output: ``[B, chunk_len, d_model]``
    Intermediate output: ``[decoder_layers, B, chunk_len, d_model]``
    """

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.config = config
        self.query_embedding = nn.Embedding(config.chunk_len, config.d_model)
        self.decoder = ACTTransformerDecoder(config)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        memory_position: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        _require_tokens(memory, "memory", self.config.d_model)
        _require_optional_position(memory_position, memory, "memory_position")
        batch_size = memory.shape[0]
        query_position = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        target = torch.zeros_like(query_position)
        return self.decoder(
            target,
            memory,
            query_position=query_position,
            memory_position=memory_position,
            memory_padding_mask=memory_padding_mask,
            return_intermediate=return_intermediate,
        )


def _reset_xavier(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)


def _require_tokens(tensor: torch.Tensor, name: str, d_model: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3 or tensor.shape[-1] != d_model:
        raise ValueError(f"{name} must have shape [B, S, {d_model}]")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _require_optional_position(
    position: Optional[torch.Tensor],
    tokens: torch.Tensor,
    name: str,
) -> None:
    if position is None:
        return
    if not isinstance(position, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if position.shape != tokens.shape:
        raise ValueError(
            f"{name} must have shape {tuple(tokens.shape)}, "
            f"got {tuple(position.shape)}"
        )
    if not position.is_floating_point():
        raise ValueError(f"{name} must be floating point")
