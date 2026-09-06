"""PyTorch architecture used only for training and ONNX export."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from ishaara_runtime import FEATURE_DIM


@dataclass
class TransformerConfig:
    size: str = "small"
    input_size: int = FEATURE_DIM
    max_position_embeddings: int = field(default=256, repr=False)
    layer_norm_eps: float = field(default=1e-12, repr=False)
    hidden_dropout_prob: float = field(default=0.1, repr=False)
    hidden_size: int = field(default=512, repr=False)
    num_attention_heads: int = field(default=8, repr=False)
    num_hidden_layers: int = field(default=4, repr=False)
    model_config: BertConfig = field(init=False)

    def __post_init__(self) -> None:
        if self.size not in {"small", "large"}:
            raise ValueError("Transformer size must be 'small' or 'large'.")
        if self.size == "small":
            self.hidden_size = 256
            self.num_attention_heads = 4
            self.num_hidden_layers = 2
        self.model_config = BertConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
        )


class PositionEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = "absolute"

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence_length = values.size()[1]
        positions = self.position_ids[:, :sequence_length]
        return self.dropout(self.LayerNorm(values + self.position_embeddings(positions)))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, n_classes: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(config.input_size, config.hidden_size)
        self.embedding = PositionEmbedding(config)
        self.layers = nn.ModuleList(BertLayer(config.model_config) for _ in range(config.num_hidden_layers))
        self.l2 = nn.Linear(config.hidden_size, n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.embedding(self.l1(values))
        for layer in self.layers:
            values = layer(values)[0]
        values = torch.max(values, dim=1).values
        values = functional.dropout(values, p=0.2, training=self.training)
        return self.l2(values)
