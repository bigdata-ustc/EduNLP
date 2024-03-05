# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CPT model. modified from transformers==4.4.1"""
import math
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm, CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import BartConfig as CPTConfig
from transformers import BertModel, BertConfig


logger = logging.get_logger(__name__)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores + attention_mask


def init_method(std):
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class CPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # CPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class CPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # with mpu.get_cuda_rng_tracker().fork():
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class CPTDecoderLayer(nn.Module):
    def __init__(self, config: CPTConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = CPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = CPTAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CPTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class CPTPretrainedModel(PreTrainedModel):
    config_class = CPTConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class CPTDecoder(CPTPretrainedModel):
    def __init__(self, config: CPTConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = CPTLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([CPTDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class CPTModel(CPTPretrainedModel):
    def __init__(self, config: CPTConfig):
        super().__init__(config)
        encoder_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.encoder_layers,
            num_attention_heads=config.encoder_attention_heads,
            intermediate_size=config.encoder_ffn_dim,
            hidden_dropout_prob=config.activation_dropout,
            attention_probs_dropout_prob=config.attention_dropout,
            max_position_embeddings=config.max_position_embeddings,
        )
        config.vocab_size = encoder_config.vocab_size
        self.encoder = BertModel(encoder_config, add_pooling_layer=False)
        self.shared = self.encoder.get_input_embeddings()
        self.decoder = CPTDecoder(config, self.shared)
        self.num_decoder_layers = config.decoder_layers
        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.set_input_embeddings(self.shared)
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        class _Encoder(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, *args, **kwargs):
                kwargs["output_hidden_states"] = True
                return self.encoder(*args, **kwargs)

        return _Encoder(self.encoder)

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, CPT automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.ones_like(input_ids),
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and isinstance(encoder_outputs, (tuple, list)):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if isinstance(encoder_outputs, (torch.Tensor)):
            encoder_hidden_states = encoder_outputs
        else:
            encoder_hidden_states = encoder_outputs[1][-self.num_decoder_layers - 1]

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # modify
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state if isinstance(encoder_outputs, dict) else None,
            # encoder_hidden_states=encoder_outputs.hidden_states if isinstance(encoder_outputs, dict) else None,
            # encoder_attentions=encoder_outputs.attentions if isinstance(encoder_outputs, dict) else None,
            encoder_last_hidden_state=encoder_hidden_states,
        )


class CPTForConditionalGeneration(CPTPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = CPTModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs=None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            device = encoder_outputs.last_hidden_state.device
            encoder_outputs["hidden_states"] = tuple(
                h.index_select(0, expanded_return_idx.to(device)) for h in encoder_outputs["hidden_states"]
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class CPTForSequenceClassification(CPTPretrainedModel):
    def __init__(self, config: CPTConfig, cls_mode=3, **kwargs):
        super().__init__(config, **kwargs)
        self.model = CPTModel(config)
        cls_mode = getattr(config, "cls_mode", cls_mode)
        if cls_mode == 1:
            logger.info("Encoder for classification.")
            cls_dim = config.d_model
        elif cls_mode == 2:
            logger.info("Decoder for classification.")
            cls_dim = config.d_model
        elif cls_mode == 3:
            logger.info("Both encoder & decoder for classification.")
            cls_dim = config.d_model * 2
        else:
            raise NotImplementedError

        self.cls_head = CPTClassificationHead(
            cls_dim,
            cls_dim,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.cls_head.dense)
        self.model._init_weights(self.cls_head.out_proj)
        self.cls_mode = cls_mode
        config.cls_mode = cls_mode

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        enc_hidden_states = outputs.encoder_last_hidden_state
        enc_rep = enc_hidden_states[:, 0]

        if self.cls_mode >= 2:
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            dec_rep = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        if self.cls_mode == 1:
            logits = self.cls_head(enc_rep)
        elif self.cls_mode == 2:
            logits = self.cls_head(dec_rep)
        elif self.cls_mode == 3:
            rep = torch.cat([enc_rep, dec_rep], dim=-1)
            logits = self.cls_head(rep)
        else:
            raise NotImplementedError

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class CPTForPretraining(CPTPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: CPTConfig):
        super().__init__(config)
        self.model = CPTModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.num_decoder_layers = config.decoder_layers

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_decoder=None,
    ):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        batch_ids = torch.arange(input_ids.size(0)).to(use_decoder)
        use_decoder_batch_ids = batch_ids[use_decoder == 1]
        no_use_decoder_batch_ids = batch_ids[use_decoder != 1]
        reorder_batch_ids = torch.cat([use_decoder_batch_ids, no_use_decoder_batch_ids], dim=0)
        input_ids = input_ids[reorder_batch_ids]
        attention_mask = attention_mask[reorder_batch_ids]
        decoder_input_ids = decoder_input_ids[reorder_batch_ids]
        num_use_decoder = use_decoder_batch_ids.size(0)

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.ones_like(input_ids),
            output_hidden_states=True,
        )
        encoder_outputs_for_decoder = encoder_outputs.hidden_states[-self.num_decoder_layers-1]
        encoder_output = encoder_outputs.last_hidden_state

        decoder_lm_logits = None
        if num_use_decoder > 0:
            decoder_outputs = self.model(
                input_ids[:num_use_decoder], attention_mask=attention_mask[:num_use_decoder],
                decoder_input_ids=decoder_input_ids[:num_use_decoder],
                encoder_outputs=encoder_outputs_for_decoder[:num_use_decoder]
            ).last_hidden_state
            decoder_lm_logits = self.lm_head(decoder_outputs) + self.final_logits_bias

        encoder_lm_logits = None
        if num_use_decoder < input_ids.size(0):
            encoder_lm_logits = self.lm_head(encoder_output[num_use_decoder:]) + self.final_logits_bias

        if decoder_lm_logits is None:
            reorder_lm_logits = encoder_lm_logits
        elif encoder_lm_logits is None:
            reorder_lm_logits = decoder_lm_logits
        else:
            reorder_lm_logits = torch.cat([decoder_lm_logits, encoder_lm_logits], dim=0)
        _, reverse_batch_ids = reorder_batch_ids.sort(dim=0)
        lm_logits = reorder_lm_logits[reverse_batch_ids]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_lm_logits,
        )


class CPTForSC(CPTPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: CPTConfig, fronzen=True, cross=False, decoder_rate=0.5):
        super().__init__(config)
        self.model = CPTModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.num_decoder_layers = config.decoder_layers
        self.fronzen = fronzen
        self.cross = cross
        self.decoder_rate = decoder_rate

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def encode_and_decode(self, input_ids, attention_mask, decoder_input_ids, num_use_decoder):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.ones_like(input_ids),
            output_hidden_states=True,
        )
        encoder_outputs_for_decoder = encoder_outputs.hidden_states[-self.num_decoder_layers-1]
        encoder_output = encoder_outputs.last_hidden_state

        decoder_lm_logits = None
        encoder_lm_logits = None

        if num_use_decoder > 0:
            decoder_outputs = self.model(
                input_ids[:num_use_decoder], attention_mask=attention_mask[:num_use_decoder],
                decoder_input_ids=decoder_input_ids[:num_use_decoder],
                encoder_outputs=encoder_outputs_for_decoder[:num_use_decoder]
            ).last_hidden_state
            decoder_lm_logits = self.lm_head(decoder_outputs) + self.final_logits_bias

        if num_use_decoder < input_ids.size(0):
            encoder_lm_logits = self.lm_head(encoder_output[num_use_decoder:]) + self.final_logits_bias

        if decoder_lm_logits is None:
            reorder_lm_logits = encoder_lm_logits
        elif encoder_lm_logits is None:
            reorder_lm_logits = decoder_lm_logits
        else:
            reorder_lm_logits = torch.cat([decoder_lm_logits, encoder_lm_logits], dim=0)
        
        return reorder_lm_logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        adv_labels=None,
        use_decoder=None,
    ):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        batch_ids = torch.arange(input_ids.size(0)).to(use_decoder)
        use_decoder_batch_ids = batch_ids[use_decoder == 1]
        no_use_decoder_batch_ids = batch_ids[use_decoder != 1]
        reorder_batch_ids = torch.cat([use_decoder_batch_ids, no_use_decoder_batch_ids], dim=0)
        input_ids = input_ids[reorder_batch_ids]
        attention_mask = attention_mask[reorder_batch_ids]
        decoder_input_ids = decoder_input_ids[reorder_batch_ids]
        labels = labels[reorder_batch_ids]
        adv_labels = adv_labels[reorder_batch_ids]
        num_use_decoder = use_decoder_batch_ids.size(0)

        loss_fct = CrossEntropyLoss()

        if self.fronzen:
            with torch.no_grad():
                lm_logits = self.encode_and_decode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    num_use_decoder=num_use_decoder,
                )
            lm_loss = None
        else:
            lm_logits = self.encode_and_decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                num_use_decoder=num_use_decoder,
            )
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        generate_input_ids = lm_logits.argmax(-1)
        not_masked_indices = (labels == -100)
        generate_input_ids[not_masked_indices] = input_ids[not_masked_indices]

        if self.cross:
            # cross the adv process
            use_decoder = torch.bernoulli(torch.tensor([self.decoder_rate]*input_ids.size(0))).long()
        else:
            # adv process need reverse the use_decoder
            use_decoder = torch.ones_like(use_decoder)
            use_decoder[:num_use_decoder] = 0
        batch_ids = torch.arange(input_ids.size(0)).to(use_decoder)
        use_decoder_batch_ids = batch_ids[use_decoder == 1]
        no_use_decoder_batch_ids = batch_ids[use_decoder != 1]
        reorder_batch_ids = torch.cat([use_decoder_batch_ids, no_use_decoder_batch_ids], dim=0)
        generate_input_ids = generate_input_ids[reorder_batch_ids]
        attention_mask = attention_mask[reorder_batch_ids]
        decoder_input_ids = decoder_input_ids[reorder_batch_ids]
        adv_labels = adv_labels[reorder_batch_ids]
        num_use_decoder = use_decoder_batch_ids.size(0)

        adv_lm_logits = self.encode_and_decode(
            input_ids=generate_input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            num_use_decoder=num_use_decoder,
        )

        adv_lm_loss = loss_fct(adv_lm_logits.view(-1, self.config.vocab_size), adv_labels.view(-1))

        if lm_loss is None:
            lm_loss = torch.zeros_like(adv_lm_loss)

        loss = adv_lm_loss + lm_loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=adv_lm_logits,
        )