import transformers
from transformers.models.llama import modeling_llama

from transformers import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, LLAMA_START_DOCSTRING, LlamaDecoderLayer, LlamaForCausalLM

import math
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

TOP_K = 40
TOP_P = 1.0
TAU = 1.0

TOP_K1 = 40
TOP_P1 = 1.0
TAU1 = 1.0

TOP_K1A = 40
TOP_P1A = 1.0
TAU1A = 1.0

def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = 10, top_p: float = 1.0
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    # logits = logits[:, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IntentionModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)
        self.action_info = dict(
            action_idx=[],
            action_prob=[],
        )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.deterministic = False
        self.tau = TAU
        self.top_k = TOP_K
        self.top_p = TOP_P

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_code = config.num_code
        self.lm_head_bias = False
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Action Extractor
        self.action_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_action_layer)])
        self.action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.action_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.action_code_book = nn.Linear(self.num_code, config.hidden_size, bias=self.lm_head_bias)
        self.action_action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.action_state_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Policy
        self.policy_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_policy_layer)])
        self.policy_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_action_info(self, action_idx):
        self.action_info["action_idx"].append(action_idx)
    
    def set_action_prob_info(self, action_prob):
        self.action_info["action_prob"].append(action_prob)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_policy, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None

        for number, decoder_layer in enumerate(self.layers):
            if policy_embeds is not None and number < 28:
                continue
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if number == 28:
                if emb_mode:
                    return hidden_states.detach()
                if policy_embeds is not None:
                    hidden_states = policy_embeds.detach()
                if vq_mode:
                    # hidden_states = hidden_states.detach()
                    hidden_states_action = hidden_states.clone()
                    hidden_states_policy = None
                elif bc_mode:
                    hidden_states = hidden_states.detach()
                    hidden_states_action = hidden_states.clone()
                    hidden_states_policy = hidden_states.clone()
                else:
                    hidden_states = hidden_states.detach()
                    hidden_states_action = None
                    hidden_states_policy = hidden_states.clone()
                
                if hidden_states_action is not None:
                    for action_layer in self.action_layers:
                        if output_hidden_states:
                            all_hidden_states_action += (hidden_states_action,)
                        if self.gradient_checkpointing and self.training:
                            layer_outputs_action = self._gradient_checkpointing_func(
                                action_layer.__call__,
                                hidden_states_action,
                                causal_mask,
                                position_ids,
                                past_key_values_action,
                                output_attentions,
                                use_cache,
                                cache_position,
                                position_embeddings,
                            )
                        else:
                            layer_outputs_action = action_layer(
                                hidden_states_action,
                                attention_mask=causal_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values_action,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                                position_embeddings=position_embeddings,
                            )
                        hidden_states_action = layer_outputs_action[0]
                        if use_cache:
                            next_decoder_cache_action = layer_outputs_action[2 if output_attentions else 1]
                        if output_attentions:
                            all_self_attns_action += (layer_outputs_action[1],)
                    hidden_states_action = self.action_norm(hidden_states_action)
                    hidden_states_action = self.action_head(hidden_states_action)

                    # hidden_states_action[:, :-1] = hidden_states_action[:, 1:].clone()
                    hidden_states_action_probs_ = F.softmax(hidden_states_action, dim=-1)

                if hidden_states_policy is not None:
                    for policy_layer in self.policy_layers:
                        if output_hidden_states:
                            all_hidden_states_policy += (hidden_states_policy,)
                        if self.gradient_checkpointing and self.training:
                            layer_outputs_policy = self._gradient_checkpointing_func(
                                policy_layer.__call__,
                                hidden_states_policy,
                                causal_mask,
                                position_ids,
                                past_key_values_policy,
                                output_attentions,
                                use_cache,
                                cache_position,
                                position_embeddings,
                            )
                        else:
                            layer_outputs_policy = policy_layer(
                                hidden_states_policy,
                                attention_mask=causal_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values_policy,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                                position_embeddings=position_embeddings,
                            )
                        hidden_states_policy = layer_outputs_policy[0]
                        if use_cache:
                            next_decoder_cache_policy = layer_outputs_policy[2 if output_attentions else 1]
                        if output_attentions:
                            all_self_attns_policy += (layer_outputs_policy[1],)
                    hidden_states_policy = self.policy_norm(hidden_states_policy)
                    hidden_states_policy_logits = self.policy_head(hidden_states_policy)
                if bc_mode:
                    hidden_states_action_probs = hidden_states_action_probs_.detach().clone()
                    # hidden_states_action_probs[:, :-1] = hidden_states_action_probs[:, 1:].clone()
                    action_idx = hidden_states_action_probs.argmax(dim=-1)
                    return hidden_states_policy_logits, action_idx
                if act_mode:
                    return hidden_states_policy_logits
                if vq_mode:
                    hidden_states_action_probs = hidden_states_action_probs_.detach().clone()
                    hidden_states_action_probs[:, :-1] = hidden_states_action_probs[:, 1:].clone()
                    action_idx = hidden_states_action_probs.argmax(dim=-1)

                    onehot = torch.eye(hidden_states_action_probs.shape[-1]).to(hidden_states_action_probs.device)
                    action_onehot = onehot[action_idx, :].type(hidden_states_action_probs.dtype)
                    hidden_states_action_emb = (action_onehot - hidden_states_action_probs).detach() + hidden_states_action_probs
                else:
                    if not self.deterministic: # TODO
                        hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
                        bs, lens, dims = hidden_states_policy_logits.shape
                        hidden_states_policy_logits_ = hidden_states_policy_logits.reshape(-1, dims)
                        policy_idx = sample(hidden_states_policy_logits_, top_k=self.top_k, temperature=self.tau, top_p=self.top_p).squeeze(-1)
                        policy_idx = policy_idx.reshape(bs, lens)
                        # hidden_states_policy_probs = F.gumbel_softmax(hidden_states_policy_logits, dim=-1, tau=1.0)
                    else:
                        hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
                        policy_idx = hidden_states_policy_probs.argmax(dim=-1)
                    
                    self.set_action_prob_info(action_prob=hidden_states_policy_probs.gather(dim=-1, index=policy_idx.unsqueeze(-1)).squeeze(-1)[:, -1:])
                    self.set_action_info(action_idx=policy_idx[:, -1:])
                    onehot = torch.eye(hidden_states_policy_logits.shape[-1]).to(hidden_states_policy_logits.device)
                    hidden_states_action_emb = onehot[policy_idx, :].type(hidden_states_policy_logits.dtype)
                
                hidden_states_action = self.action_code_book(hidden_states_action_emb)
                hidden_states = self.action_action_norm(hidden_states_action) + self.action_state_norm(hidden_states)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values_main,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values_main,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        next_cache_policy = next_decoder_cache_policy if use_cache else None
        next_cache_action = None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
            next_cache_policy = next_cache_policy.to_legacy_cache()
            next_cache = {
                "main": next_cache,
                "policy": next_cache_policy,
            }

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class IntentionForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if hasattr(self.config, "v"):
            if self.config.v == 1:
                self.model = IntentionModel_v1(config)
            elif self.config.v == 2:
                self.model = IntentionModel_v1p(config)
            elif self.config.v == 0:
                # print("\n\n\n\n version 1a \n\n\n\n\n")
                self.model = IntentionModel_v1a(config)
            else:
                self.model = IntentionModel(config)
        else:
            self.model = IntentionModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    # def add_policy_layer(self, add_num=8):
    #     self.model.add_policy_layer(add_num)
    #     self.config.num_policy_layer += add_num
    #     self.config.num_add_policy_layer = 0
    def set_action_sampling(self, greedy=False, tau=1.0, top_k=50, top_p=1.0):
        self.model.deterministic = greedy
        self.model.tau = tau
        self.model.top_k = top_k
        self.model.top_p = top_p
    
    def reset_action_info(self):
        for key, _ in self.model.action_info.items():
            self.model.action_info[key].clear()
    
    def get_action_info(self, prob=False):
        action_info = torch.cat(self.model.action_info["action_idx"], dim=1)
        if prob:
            return action_info, torch.cat(self.model.action_info["action_prob"], dim=1)
        # action_info = torch.cat(self.model.action_info["action_idx"], dim=1)
        return action_info

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        ea_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            vq_mode=vq_mode,
            bc_mode=bc_mode,
            act_mode=act_mode,
            emb_mode=emb_mode,
            ea_mode=ea_mode,
            policy_embeds=policy_embeds,
        )
        if act_mode or bc_mode or emb_mode or ea_mode:
            return outputs

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def forward_world_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
        action_idx: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward_world_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            vq_mode=vq_mode,
            bc_mode=bc_mode,
            act_mode=act_mode,
            emb_mode=emb_mode,
            policy_embeds=policy_embeds,
            action_idx=action_idx,
        )
        logits = self.lm_head(outputs)
        logits = logits.float()
        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if past_key_values is not None and isinstance(past_key_values, Dict) and isinstance(past_key_values['main'], StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values['main'].get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class LlamaMergeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size * 2, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size * 2, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, x_a):
        x = torch.cat([x, x_a], dim=-1)
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IntentionModel_v1(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.eval_fixed_action_idx = None
        self.action_info = dict(
            action_idx=[],
            action_prob=[],
        )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.deterministic = False
        self.tau = TAU1
        self.top_k = TOP_K1
        self.top_p = TOP_P1

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_code = config.num_code
        self.lm_head_bias = False
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Action Extractor
        self.action_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_action_layer)])
        self.action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.action_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.action_code_book = nn.Linear(self.num_code, config.hidden_size, bias=self.lm_head_bias)
        self.action_merge_layers = nn.ModuleList([LlamaMergeMLP(config) for _ in range(config.num_dyna_layer)])
        # self.action_action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.action_state_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Policy
        self.policy_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_policy_layer)])
        self.policy_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.gradient_checkpointing = False
        self.intention_model_config = config
        self.start_index = None

        # Initialize weights and apply final processing
        self.post_init()
    
    # def set_start_index(self, index):
    #     self.start_index = index
    #     self.add_num = self.intention_model_config.num_hidden_layers - index

    # def add_policy_layer(self, add_num):
    #     self.add_num = add_num
    #     additional_policy_layers = nn.ModuleList([LlamaDecoderLayer(self.intention_model_config, layer_idx) for layer_idx in range(add_num)])
    #     start_index = self.intention_model_config.num_hidden_layers - add_num
    #     for idx, layer in enumerate(additional_policy_layers):
    #         layer.load_state_dict(self.layers[start_index + idx].state_dict())
    #     self.policy_layers = additional_policy_layers + self.policy_layers
    #     # self.start_index = start_index
    #     self.set_start_index(start_index)
    
    def set_action_info(self, action_idx):
        self.action_info["action_idx"].append(action_idx)

    def set_action_prob_info(self, action_prob):
        self.action_info["action_prob"].append(action_prob)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        ea_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_policy, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None
        hidden_states_policy = None

        if policy_embeds is None:
            for number, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # if self.start_index is not None and self.start_index == number:
                #     hidden_states_policy = hidden_states.clone().detach()
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values_main,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_main,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        else:
            hidden_states = policy_embeds  # .detach()

        if emb_mode and not ea_mode:
            return hidden_states.detach()
        if vq_mode:
            # hidden_states = hidden_states.detach()
            hidden_states_action = hidden_states.clone()
            hidden_states_policy = None
        elif bc_mode:
            hidden_states = hidden_states.detach()
            hidden_states_action = hidden_states.clone()
            hidden_states_policy = hidden_states.clone() # if hidden_states_policy is None else hidden_states_policy
        else:
            if not self.deterministic:
                hidden_states = hidden_states  #.detach()
            hidden_states_action = None
            hidden_states_policy = hidden_states.clone()
        
        if hidden_states_action is not None:
            for action_layer in self.action_layers:
                if output_hidden_states:
                    all_hidden_states_action += (hidden_states_action,)
                if self.gradient_checkpointing and self.training:
                    layer_outputs_action = self._gradient_checkpointing_func(
                        action_layer.__call__,
                        hidden_states_action,
                        causal_mask,
                        position_ids,
                        past_key_values_action,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs_action = action_layer(
                        hidden_states_action,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_action,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states_action = layer_outputs_action[0]
                if use_cache:
                    next_decoder_cache_action = layer_outputs_action[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns_action += (layer_outputs_action[1],)
            hidden_states_action = self.action_norm(hidden_states_action)
            hidden_states_action = self.action_head(hidden_states_action)

            # hidden_states_action[:, :-1] = hidden_states_action[:, 1:].clone()
            hidden_states_action_probs_ = F.softmax(hidden_states_action, dim=-1)

        if hidden_states_policy is not None:
            for num_pi, policy_layer in enumerate(self.policy_layers):
                if output_hidden_states:
                    all_hidden_states_policy += (hidden_states_policy,)
                # if self.start_index is not None and num_pi == self.add_num:
                #     hidden_states_policy = self.norm(hidden_states_policy)
                #     print("Policy N {}, remaining {}".format(num_pi, len(self.policy_layers) - num_pi))
                if self.gradient_checkpointing and self.training:
                    layer_outputs_policy = self._gradient_checkpointing_func(
                        policy_layer.__call__,
                        hidden_states_policy,
                        causal_mask,
                        position_ids,
                        past_key_values_policy,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs_policy = policy_layer(
                        hidden_states_policy,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_policy,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states_policy = layer_outputs_policy[0]
                if use_cache:
                    next_decoder_cache_policy = layer_outputs_policy[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns_policy += (layer_outputs_policy[1],)
            hidden_states_policy = self.policy_norm(hidden_states_policy)
            hidden_states_policy_logits = self.policy_head(hidden_states_policy)
        
        if ea_mode:
            if use_cache:
                next_cache = next_decoder_cache if use_cache else None
                next_cache_policy = next_decoder_cache_policy if use_cache else None
                next_cache_action = None
                if return_legacy_cache:
                    next_cache = next_cache.to_legacy_cache()
                    next_cache_policy = next_cache_policy.to_legacy_cache()
                    next_cache = {
                        "main": next_cache,
                        "policy": next_cache_policy,
                    }
                return [hidden_states_policy_logits / self.tau, next_cache, hidden_states.detach()]
            return [hidden_states_policy_logits / self.tau, hidden_states.detach()]
        
        if bc_mode:
            hidden_states_action_probs = hidden_states_action_probs_.detach().clone()
            # hidden_states_action_probs[:, :-1] = hidden_states_action_probs[:, 1:].clone()
            action_idx = hidden_states_action_probs.argmax(dim=-1)
            return hidden_states_policy_logits, action_idx
        if act_mode:
            return hidden_states_policy_logits / self.tau
        if vq_mode:
            hidden_states_action_probs = hidden_states_action_probs_.detach().clone()
            hidden_states_action_probs[:, :-1] = hidden_states_action_probs[:, 1:].clone()
            action_idx = hidden_states_action_probs.argmax(dim=-1)

            onehot = torch.eye(hidden_states_action_probs.shape[-1]).to(hidden_states_action_probs.device)
            action_onehot = onehot[action_idx, :].type(hidden_states_action_probs.dtype)
            hidden_states_action_emb = (action_onehot - hidden_states_action_probs).detach() + hidden_states_action_probs
        else:
            if not self.deterministic: # TODO
                hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
                bs, lens, dims = hidden_states_policy_logits.shape
                hidden_states_policy_logits_ = hidden_states_policy_logits.reshape(-1, dims)
                policy_idx = sample(hidden_states_policy_logits_, top_k=self.top_k, temperature=self.tau, top_p=self.top_p).squeeze(-1)
                policy_idx = policy_idx.reshape(bs, lens)
                # hidden_states_policy_probs = F.gumbel_softmax(hidden_states_policy_logits, dim=-1, tau=1.0)
            else:
                hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
                policy_idx = hidden_states_policy_probs.argmax(dim=-1)
            
            self.set_action_prob_info(action_prob=hidden_states_policy_probs.gather(dim=-1, index=policy_idx.unsqueeze(-1)).squeeze(-1)[:, -1:])
            self.set_action_info(action_idx=policy_idx[:, -1:])
            
            if self.eval_fixed_action_idx is not None:
                onehot = torch.eye(hidden_states_policy_logits.shape[-1]).to(hidden_states_policy_logits.device)
                hidden_states_action_emb = onehot[self.eval_fixed_action_idx, :].type(hidden_states_policy_logits.dtype).unsqueeze(0).unsqueeze(0).repeat(policy_idx.shape[0], policy_idx.shape[1], 1)
            else:
                onehot = torch.eye(hidden_states_policy_logits.shape[-1]).to(hidden_states_policy_logits.device)
                hidden_states_action_emb = onehot[policy_idx, :].type(hidden_states_policy_logits.dtype)
        
        hidden_states_action = self.action_code_book(hidden_states_action_emb)
        hidden_states_state = hidden_states.clone()
        for block_merge in self.action_merge_layers:
            hidden_states_state = block_merge(hidden_states_state, hidden_states_action) + hidden_states_state
        if not self.deterministic:
            hidden_states = self.norm(hidden_states_state) - hidden_states.detach()
        else:
            hidden_states = self.norm(hidden_states_state).detach() - hidden_states
        # hidden_states = self.norm(hidden_states_state) - hidden_states

        next_cache = next_decoder_cache if use_cache else None
        next_cache_policy = next_decoder_cache_policy if use_cache else None
        next_cache_action = None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
            next_cache_policy = next_cache_policy.to_legacy_cache()
            next_cache = {
                "main": next_cache,
                "policy": next_cache_policy,
            }

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def forward_world_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
        action_idx: torch.LongTensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                # "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            # past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_main, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                # past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            # past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None
        hidden_states_policy = None

        if policy_embeds is None:
            for number, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # if self.start_index is not None and self.start_index == number:
                #     hidden_states_policy = hidden_states.clone().detach()
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values_main,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_main,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        else:
            hidden_states = policy_embeds.detach()

        onehot = torch.eye(self.num_code).to(hidden_states.device)
        hidden_states_action_emb = onehot[action_idx, :].type(hidden_states.dtype).unsqueeze(0).unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)  # bs, 1, 1
        
        hidden_states_last = hidden_states.clone()[:, -1:]
        hidden_states_action = self.action_code_book(hidden_states_action_emb)
        hidden_states_state = hidden_states_last.clone()
        for block_merge in self.action_merge_layers:
            hidden_states_state = block_merge(hidden_states_state, hidden_states_action) + hidden_states_state
        
        hidden_states = self.norm(hidden_states_state) - hidden_states_last

        return hidden_states

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    



@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IntentionModel_v1p(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.eval_fixed_action_idx = None
        self.action_info = dict(
            action_idx=[],
            action_prob=[],
        )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.deterministic = False
        self.tau = TAU1
        self.top_k = TOP_K1
        self.top_p = TOP_P1

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_code = config.num_code
        self.lm_head_bias = False
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Action Extractor
        self.action_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_action_layer)])
        self.action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.action_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.action_code_book = nn.Linear(self.num_code, config.hidden_size, bias=self.lm_head_bias)
        self.action_merge_layers = nn.ModuleList([LlamaMergeMLP(config) for _ in range(config.num_dyna_layer)])
        # self.action_action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.action_state_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Policy
        self.policy_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_policy_layer)])
        self.policy_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.gradient_checkpointing = False
        self.intention_model_config = config
        self.start_index = None

        # Initialize weights and apply final processing
        self.post_init()
    
    # def set_start_index(self, index):
    #     self.start_index = index
    #     self.add_num = self.intention_model_config.num_hidden_layers - index

    # def add_policy_layer(self, add_num):
    #     self.add_num = add_num
    #     additional_policy_layers = nn.ModuleList([LlamaDecoderLayer(self.intention_model_config, layer_idx) for layer_idx in range(add_num)])
    #     start_index = self.intention_model_config.num_hidden_layers - add_num
    #     for idx, layer in enumerate(additional_policy_layers):
    #         layer.load_state_dict(self.layers[start_index + idx].state_dict())
    #     self.policy_layers = additional_policy_layers + self.policy_layers
    #     # self.start_index = start_index
    #     self.set_start_index(start_index)
    
    def set_action_info(self, action_idx):
        self.action_info["action_idx"].append(action_idx)

    def set_action_prob_info(self, action_prob):
        self.action_info["action_prob"].append(action_prob)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        ea_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_policy, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None
        hidden_states_policy = None

        if policy_embeds is None:
            for number, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # if self.start_index is not None and self.start_index == number:
                #     hidden_states_policy = hidden_states.clone().detach()
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values_main,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_main,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        else:
            hidden_states = policy_embeds  # .detach()

        if emb_mode and not ea_mode:
            return hidden_states.detach()
        if vq_mode:
            # hidden_states = hidden_states.detach()
            hidden_states_action = hidden_states.clone()
            hidden_states_policy = None
        elif bc_mode:
            hidden_states = hidden_states.detach()
            hidden_states_action = hidden_states.clone()
            hidden_states_policy = hidden_states.clone() # if hidden_states_policy is None else hidden_states_policy
        else:
            # if self.deterministic:
            #     hidden_states = hidden_states.detach()  #.detach()
            hidden_states_action = None
            hidden_states_policy = hidden_states.clone()
        
        if hidden_states_action is not None:
            for action_layer in self.action_layers:
                if output_hidden_states:
                    all_hidden_states_action += (hidden_states_action,)
                if self.gradient_checkpointing and self.training:
                    layer_outputs_action = self._gradient_checkpointing_func(
                        action_layer.__call__,
                        hidden_states_action,
                        causal_mask,
                        position_ids,
                        past_key_values_action,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs_action = action_layer(
                        hidden_states_action,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_action,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states_action = layer_outputs_action[0]
                if use_cache:
                    next_decoder_cache_action = layer_outputs_action[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns_action += (layer_outputs_action[1],)
            hidden_states_action = self.action_norm(hidden_states_action)
            hidden_states_action = self.action_head(hidden_states_action)

            # hidden_states_action[:, :-1] = hidden_states_action[:, 1:].clone()
            hidden_states_action_probs_ = F.softmax(hidden_states_action, dim=-1)

        if hidden_states_policy is not None:
            for num_pi, policy_layer in enumerate(self.policy_layers):
                if output_hidden_states:
                    all_hidden_states_policy += (hidden_states_policy,)
                # if self.start_index is not None and num_pi == self.add_num:
                #     hidden_states_policy = self.norm(hidden_states_policy)
                #     print("Policy N {}, remaining {}".format(num_pi, len(self.policy_layers) - num_pi))
                if self.gradient_checkpointing and self.training:
                    layer_outputs_policy = self._gradient_checkpointing_func(
                        policy_layer.__call__,
                        hidden_states_policy,
                        causal_mask,
                        position_ids,
                        past_key_values_policy,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs_policy = policy_layer(
                        hidden_states_policy,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_policy,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states_policy = layer_outputs_policy[0]
                if use_cache:
                    next_decoder_cache_policy = layer_outputs_policy[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns_policy += (layer_outputs_policy[1],)
            hidden_states_policy = self.policy_norm(hidden_states_policy)
            hidden_states_policy_logits = self.policy_head(hidden_states_policy)
        
        if ea_mode:
            if use_cache:
                next_cache = next_decoder_cache if use_cache else None
                next_cache_policy = next_decoder_cache_policy if use_cache else None
                next_cache_action = None
                if return_legacy_cache:
                    next_cache = next_cache.to_legacy_cache()
                    next_cache_policy = next_cache_policy.to_legacy_cache()
                    next_cache = {
                        "main": next_cache,
                        "policy": next_cache_policy,
                    }
                return [hidden_states_policy_logits / self.tau, next_cache, hidden_states.detach()]
            return [hidden_states_policy_logits / self.tau, hidden_states.detach()]
        
        if bc_mode:
            hidden_states_action_probs = hidden_states_action_probs_.detach().clone()
            # hidden_states_action_probs[:, :-1] = hidden_states_action_probs[:, 1:].clone()
            action_idx = hidden_states_action_probs.argmax(dim=-1)
            return hidden_states_policy_logits, action_idx
        if act_mode:
            return hidden_states_policy_logits / self.tau
        if vq_mode:
            hidden_states_action_probs = hidden_states_action_probs_.detach().clone()
            hidden_states_action_probs[:, :-1] = hidden_states_action_probs[:, 1:].clone()
            action_idx = hidden_states_action_probs.argmax(dim=-1)

            onehot = torch.eye(hidden_states_action_probs.shape[-1]).to(hidden_states_action_probs.device)
            action_onehot = onehot[action_idx, :].type(hidden_states_action_probs.dtype)
            hidden_states_action_emb = (action_onehot - hidden_states_action_probs).detach() + hidden_states_action_probs
        else:
            if not self.deterministic: # TODO
                hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
                bs, lens, dims = hidden_states_policy_logits.shape
                hidden_states_policy_logits_ = hidden_states_policy_logits.reshape(-1, dims)
                policy_idx = sample(hidden_states_policy_logits_, top_k=self.top_k, temperature=self.tau, top_p=self.top_p).squeeze(-1)
                policy_idx = policy_idx.reshape(bs, lens)
                # hidden_states_policy_probs = F.gumbel_softmax(hidden_states_policy_logits, dim=-1, tau=1.0)
            else:
                hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
                policy_idx = hidden_states_policy_probs.argmax(dim=-1)
            
            self.set_action_prob_info(action_prob=hidden_states_policy_probs.gather(dim=-1, index=policy_idx.unsqueeze(-1)).squeeze(-1)[:, -1:])
            self.set_action_info(action_idx=policy_idx[:, -1:])
            
            if self.eval_fixed_action_idx is not None:
                onehot = torch.eye(hidden_states_policy_logits.shape[-1]).to(hidden_states_policy_logits.device)
                hidden_states_action_emb = onehot[self.eval_fixed_action_idx, :].type(hidden_states_policy_logits.dtype).unsqueeze(0).unsqueeze(0).repeat(policy_idx.shape[0], policy_idx.shape[1], 1)
            else:
                onehot = torch.eye(hidden_states_policy_logits.shape[-1]).to(hidden_states_policy_logits.device)
                hidden_states_action_emb = onehot[policy_idx, :].type(hidden_states_policy_logits.dtype)
        
        hidden_states_action = self.action_code_book(hidden_states_action_emb)
        hidden_states_state = hidden_states.clone()
        for block_merge in self.action_merge_layers:
            hidden_states_state = block_merge(hidden_states_state, hidden_states_action) + hidden_states_state
        # if not self.deterministic:
        #     hidden_states = self.norm(hidden_states_state) - hidden_states.detach()
        # else:
        #     hidden_states = self.norm(hidden_states_state).detach() - hidden_states
        hidden_states = self.norm(hidden_states_state) - hidden_states.detach()

        next_cache = next_decoder_cache if use_cache else None
        next_cache_policy = next_decoder_cache_policy if use_cache else None
        next_cache_action = None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
            next_cache_policy = next_cache_policy.to_legacy_cache()
            next_cache = {
                "main": next_cache,
                "policy": next_cache_policy,
            }

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def forward_world_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
        action_idx: torch.LongTensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                # "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            # past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_main, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                # past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            # past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None
        hidden_states_policy = None

        if policy_embeds is None:
            for number, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # if self.start_index is not None and self.start_index == number:
                #     hidden_states_policy = hidden_states.clone().detach()
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values_main,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_main,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        else:
            hidden_states = policy_embeds.detach()

        onehot = torch.eye(self.num_code).to(hidden_states.device)
        hidden_states_action_emb = onehot[action_idx, :].type(hidden_states.dtype).unsqueeze(0).unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)  # bs, 1, 1
        
        hidden_states_last = hidden_states.clone()[:, -1:]
        hidden_states_action = self.action_code_book(hidden_states_action_emb)
        hidden_states_state = hidden_states_last.clone()
        for block_merge in self.action_merge_layers:
            hidden_states_state = block_merge(hidden_states_state, hidden_states_action) + hidden_states_state
        
        hidden_states = self.norm(hidden_states_state) - hidden_states_last

        return hidden_states

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    



@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IntentionModel_v1a(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)
        self.action_info = dict(
            action_idx=[],
            action_prob=[],
        )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.deterministic = False
        self.tau = TAU1A
        self.top_k = TOP_K1A
        self.top_p = TOP_P1A

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_code = config.num_code
        self.lm_head_bias = False
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # # Action Extractor
        # self.action_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_action_layer)])
        # self.action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.action_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.action_code_book = nn.Linear(self.num_code, config.hidden_size, bias=self.lm_head_bias)
        self.action_merge_layers = nn.ModuleList([LlamaMergeMLP(config) for _ in range(config.num_dyna_layer)])
        # self.action_action_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.action_state_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Policy
        self.policy_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_policy_layer)])
        self.policy_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)

        self.gradient_checkpointing = False
        self.intention_model_config = config
        self.start_index = None

        # Initialize weights and apply final processing
        self.post_init()
    
    # def set_start_index(self, index):
    #     self.start_index = index
    #     self.add_num = self.intention_model_config.num_hidden_layers - index

    # def add_policy_layer(self, add_num):
    #     self.add_num = add_num
    #     additional_policy_layers = nn.ModuleList([LlamaDecoderLayer(self.intention_model_config, layer_idx) for layer_idx in range(add_num)])
    #     start_index = self.intention_model_config.num_hidden_layers - add_num
    #     for idx, layer in enumerate(additional_policy_layers):
    #         layer.load_state_dict(self.layers[start_index + idx].state_dict())
    #     self.policy_layers = additional_policy_layers + self.policy_layers
    #     # self.start_index = start_index
    #     self.set_start_index(start_index)
    
    def set_action_info(self, action_idx):
        self.action_info["action_idx"].append(action_idx)
    
    def set_action_prob_info(self, action_prob):
        self.action_info["action_prob"].append(action_prob)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        ea_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_policy, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None
        hidden_states_policy = None

        if policy_embeds is None:
            for number, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # if self.start_index is not None and self.start_index == number:
                #     hidden_states_policy = hidden_states.clone().detach()
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values_main,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_main,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        else:
            hidden_states = policy_embeds.detach()

        if emb_mode and not ea_mode:
            return hidden_states.detach()
        if vq_mode:
            # hidden_states = hidden_states.detach()
            hidden_states_action = None  # hidden_states.clone()
            hidden_states_policy = hidden_states.detach().clone()
        else:
            # if not act_mode:
            hidden_states = hidden_states  #.detach()
            hidden_states_action = None
            hidden_states_policy = hidden_states.clone()
        assert hidden_states_policy is not None

        if hidden_states_policy is not None:
            for num_pi, policy_layer in enumerate(self.policy_layers):
                if output_hidden_states:
                    all_hidden_states_policy += (hidden_states_policy,)
                # if self.start_index is not None and num_pi == self.add_num:
                #     hidden_states_policy = self.norm(hidden_states_policy)
                #     print("Policy N {}, remaining {}".format(num_pi, len(self.policy_layers) - num_pi))
                if self.gradient_checkpointing and self.training:
                    layer_outputs_policy = self._gradient_checkpointing_func(
                        policy_layer.__call__,
                        hidden_states_policy,
                        causal_mask,
                        position_ids,
                        past_key_values_policy,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs_policy = policy_layer(
                        hidden_states_policy,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_policy,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states_policy = layer_outputs_policy[0]
                if use_cache:
                    next_decoder_cache_policy = layer_outputs_policy[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns_policy += (layer_outputs_policy[1],)
            hidden_states_policy = self.policy_norm(hidden_states_policy)
            hidden_states_policy_logits = self.policy_head(hidden_states_policy)
            # hidden_states_action_probs_ = F.softmax(hidden_states_policy_logits, dim=-1)
        
        if ea_mode:
            if use_cache:
                next_cache = next_decoder_cache if use_cache else None
                next_cache_policy = next_decoder_cache_policy if use_cache else None
                next_cache_action = None
                if return_legacy_cache:
                    next_cache = next_cache.to_legacy_cache()
                    next_cache_policy = next_cache_policy.to_legacy_cache()
                    next_cache = {
                        "main": next_cache,
                        "policy": next_cache_policy,
                    }
                return [hidden_states_policy_logits / self.tau, next_cache, hidden_states.detach()]
            return [hidden_states_policy_logits / self.tau, hidden_states.detach()]
        if act_mode:
            return hidden_states_policy_logits / self.tau

        if not self.deterministic: # TODO
            hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
            bs, lens, dims = hidden_states_policy_logits.shape
            hidden_states_policy_logits_ = hidden_states_policy_logits.reshape(-1, dims)
            policy_idx = sample(hidden_states_policy_logits_, top_k=self.top_k, temperature=self.tau, top_p=self.top_p).squeeze(-1)
            policy_idx = policy_idx.reshape(bs, lens)
            # hidden_states_policy_probs = F.gumbel_softmax(hidden_states_policy_logits, dim=-1, tau=1.0)
        else:
            hidden_states_policy_probs = F.softmax(hidden_states_policy_logits, dim=-1)
            policy_idx = hidden_states_policy_probs.argmax(dim=-1)

        self.set_action_prob_info(action_prob=hidden_states_policy_probs.gather(dim=-1, index=policy_idx.unsqueeze(-1)).squeeze(-1)[:, -1:])
        self.set_action_info(action_idx=policy_idx[:, -1:])
        onehot = torch.eye(hidden_states_policy_logits.shape[-1]).to(hidden_states_policy_logits.device)
        hidden_states_action_emb = onehot[policy_idx, :].type(hidden_states_policy_logits.dtype)
        
        hidden_states_action = self.action_code_book(hidden_states_action_emb)
        hidden_states_state = hidden_states.clone()
        for block_merge in self.action_merge_layers:
            hidden_states_state = block_merge(hidden_states_state, hidden_states_action) + hidden_states_state
        # if not self.deterministic:
        #     hidden_states = self.norm(hidden_states_state) - hidden_states.detach()
        # else:
        #     hidden_states = self.norm(hidden_states_state).detach() - hidden_states
        hidden_states = self.norm(hidden_states_state) - hidden_states

        next_cache = next_decoder_cache if use_cache else None
        next_cache_policy = next_decoder_cache_policy if use_cache else None
        next_cache_action = None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
            next_cache_policy = next_cache_policy.to_legacy_cache()
            next_cache = {
                "main": next_cache,
                "policy": next_cache_policy,
            }

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def forward_world_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vq_mode: Optional[bool] = False,
        bc_mode: Optional[bool] = False,
        act_mode: Optional[bool] = False,
        emb_mode: Optional[bool] = False,
        policy_embeds: Optional[torch.FloatTensor] = None,
        action_idx: torch.LongTensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None and policy_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and policy_embeds is not None:
            inputs_embeds = policy_embeds

        return_legacy_cache = False
        past_key_values_action = None
        if (
            use_cache and not isinstance(past_key_values, Dict) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_main = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = {
                "main": past_key_values_main,
                # "main_policy": past_key_values_main_policy,
                # "policy": past_key_values_policy,
            }
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and isinstance(past_key_values, Dict) and not self.training:
            past_key_values_main = past_key_values['main']
            # past_key_values_main_policy = past_key_values['main_policy']
            # past_key_values_policy = past_key_values['policy']
            if use_cache and not isinstance(past_key_values_main, Cache) and not self.training:
                return_legacy_cache = True
                past_key_values_main = DynamicCache.from_legacy_cache(past_key_values_main)
                # past_key_values_main_policy = DynamicCache.from_legacy_cache(past_key_values_main_policy)
                # past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values_policy)
        else:
            past_key_values_main = None
            # past_key_values_policy = None
            # past_key_values_main_policy = None

        if cache_position is None:
            past_seen_tokens = past_key_values_main.get_seq_length() if past_key_values_main is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values_main, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_action = () if output_hidden_states else None
        all_hidden_states_policy = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_attns_action = () if output_attentions else None
        all_self_attns_policy = () if output_attentions else None
        next_decoder_cache = None
        hidden_states_policy = None

        if policy_embeds is None:
            for number, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values_main,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_main,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        else:
            hidden_states = policy_embeds.detach()

        onehot = torch.eye(self.num_code).to(hidden_states.device)
        hidden_states_action_emb = onehot[action_idx, :].type(hidden_states.dtype).unsqueeze(0).unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)  # bs, 1, 1
        
        hidden_states_last = hidden_states.clone()[:, -1:]
        hidden_states_action = self.action_code_book(hidden_states_action_emb)
        hidden_states_state = hidden_states_last.clone()
        for block_merge in self.action_merge_layers:
            hidden_states_state = block_merge(hidden_states_state, hidden_states_action) + hidden_states_state
        
        hidden_states = self.norm(hidden_states_state) - hidden_states_last

        return hidden_states

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
