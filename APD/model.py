from transformers.pytorch_utils import Conv1D
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional, Tuple, Dict, Any
from .config import APDConfig
from processor import APDProcessor
from data import APDLMHeadModelOutput, APDModelOutput, APDProcessorOutput

from transformers.models.vit.modeling_vit import ViTPatchEmbeddings
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model, GPT2Attention, GPT2Config
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)


import torch.nn as nn
import torch


import math
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as OriginalGPT2Attention


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        self.c_fc = Conv1D(inner_dim, config.hidden_size)
        self.c_proj = Conv1D(config.hidden_size, inner_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


class CustomGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"Embedding dimension ({self.embed_dim}) must be divisible by number of heads ({self.num_heads})."
            )

        self.scale_attn_weights = True

        # Initialize c_attn as Conv1D
        if is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Debug print
        print(f"Input hidden_states shape: {hidden_states.shape}")

        if self.is_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError(
                    "encoder_hidden_states must be provided for cross-attention.")
            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            key, value = key_value.split(self.embed_dim, dim=2)
        else:
            qkv = self.c_attn(hidden_states)
            print(f"QKV shape after c_attn: {qkv.shape}")  # Debug print
            query, key, value = qkv.split(self.embed_dim, dim=2)

        # Debug print
        print(
            f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Debug print
        print(
            f"After split_heads - Query: {query.shape}, Key: {key.shape}, Value: {value.shape}")

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask)

        # Debug print
        print(f"After attention - attn_output shape: {attn_output.shape}")

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim)
        # Debug print
        print(f"After merge_heads - attn_output shape: {attn_output.shape}")

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        print(f"Final attn_output shape: {attn_output.shape}")  # Debug print

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @staticmethod
    def _split_heads(tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

class EnhancedGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        self.mlp = GPT2MLP(config)

        if config.add_cross_attention:
            self.ln_cross_attn = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_epsilon)
            self.crossattention = CustomGPT2Attention(
                config, is_cross_attention=True)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]  # present, (attentions)

        hidden_states = residual + attn_output

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states=hidden_states,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attn_outputs[2:]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # hidden_states, present, (attentions, cross_attentions)
        return outputs


class DynamicFeatureFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_size, num_heads=8)
        self.linear = nn.Linear(config.hidden_size * 3, 3)  # 输出改为3通道

    def forward(self, features):
        # 假设 features 的形状是 [batch_size, 3 * hidden_size, height, width]
        batch_size, channels, height, width = features.shape
        hidden_size = channels // 3

        # 将特征重塑为 [batch_size, 3, hidden_size, height, width]
        features = features.view(batch_size, 3, hidden_size, height, width)

        # 将特征重塑为 [3, batch_size * height * width, hidden_size]
        features_3d = features.permute(
            1, 0, 3, 4, 2).reshape(3, -1, hidden_size)

        # 应用自注意力
        fused_features, _ = self.attention(
            features_3d, features_3d, features_3d)

        # 将融合后的特征重塑回 [batch_size, 3 * hidden_size, height, width]
        fused_features = fused_features.reshape(
            3, batch_size, height, width, hidden_size)
        fused_features = fused_features.permute(
            1, 0, 4, 2, 3).reshape(batch_size, -1, height, width)

        # 使用线性层将通道数减少到3
        # [batch_size, height, width, 3 * hidden_size]
        fused_features = fused_features.permute(0, 2, 3, 1)
        # [batch_size, height, width, 3]
        fused_features = self.linear(fused_features)
        fused_features = fused_features.permute(
            0, 3, 1, 2)  # [batch_size, 3, height, width]

        return fused_features


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scales = [0.5, 1, 2]  # 多个尺度
        self.convs = nn.ModuleList([
            nn.Conv2d(3, config.hidden_size, kernel_size=3, padding=1)
            for _ in self.scales
        ])

    def forward(self, x):
        features = []
        for scale, conv in zip(self.scales, self.convs):
            scaled_x = F.interpolate(
                x, scale_factor=scale, mode='bilinear', align_corners=False)
            features.append(conv(scaled_x))
        return torch.cat([F.interpolate(f, size=x.shape[2:], mode='bilinear', align_corners=False)
                          for f in features], dim=1)


class APDModel(nn.Module):
    def __init__(self, config: APDConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # Initialize embed_dim here
        # embeddings
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        self.use_dynamic_fusion = config.use_dynamic_fusion
        self.cross_attention_layers = config.cross_attention_layers
        self.use_multi_scale_features = config.use_multi_scale_features

        if self.use_multi_scale_features:
            self.multi_scale_feature_extractor = MultiScaleFeatureExtractor(
                config)

        if self.use_dynamic_fusion:
            self.feature_fusion = DynamicFeatureFusion(config)

        self.gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.n_inner if hasattr(config, 'n_inner') else None,
            activation_function=config.activation_function if hasattr(
                config, 'activation_function') else "gelu_new",
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range if hasattr(
                config, 'initializer_range') else 0.02,
            scale_attn_weights=config.scale_attn_weights if hasattr(
                config, 'scale_attn_weights') else True,
            use_cache=True
        )

        self.hidden_layers = nn.ModuleList([
            EnhancedGPT2Block(config.gpt2_config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.dropout = nn.Dropout(config.attn_pdrop)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)

        self._attn_implementation = config._attn_implementation

        # initialise GPT-2 weights from Hugging Face
        self.initialise_weights(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
    ) -> APDModelOutput:
        device = input_ids.device if input_ids is not None else pixel_values.device
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        # Determine past_length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)

        if self.use_multi_scale_features:
            multi_scale_features = self.multi_scale_feature_extractor(
                pixel_values)
            if self.use_dynamic_fusion:
                fused_features = self.feature_fusion(multi_scale_features)
            else:
                fused_features = multi_scale_features.view(
                    multi_scale_features.shape[0], 3, -1,
                    multi_scale_features.shape[2], multi_scale_features.shape[3]
                ).mean(dim=2)
            patch_embeddings = self.patch_embeddings(fused_features)
        else:
            multi_scale_features = None
            patch_embeddings = self.patch_embeddings(pixel_values)

        token_embeddings = self.token_embedding(input_ids)

        if patch_embeddings is not None:
            patch_and_token_embeddings = torch.cat(
                [patch_embeddings, token_embeddings], dim=1)
        else:
            patch_and_token_embeddings = token_embeddings
        input_shape = patch_and_token_embeddings.size()

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape[0], -1)

        position_embeddings = self.positional_embedding(position_ids)

        hidden_states = patch_and_token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            # Ensure the attention mask is expanded to the right shape
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (attention_mask.shape[0], patch_embeddings.shape[1]
                         if patch_embeddings is not None else past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    ),
                    attention_mask
                ],
                dim=-1
            )

            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=input_shape[:2],
                    inputs_embeds=patch_and_token_embeddings,
                    past_key_values_length=past_length,
                )

        presents = () if use_cache else None

        for i, (hidden_layer, layer_past) in enumerate(zip(self.hidden_layers, past_key_values or [None] * len(self.hidden_layers))):
            if isinstance(hidden_layer, EnhancedGPT2Block):
                encoder_hidden_states = multi_scale_features if i in self.cross_attention_layers else None
                layer_outputs = hidden_layer(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,  # Provide if necessary
                )
        else:
            layer_outputs = hidden_layer(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
        hidden_states = layer_outputs[0]
        if use_cache:
            presents = presents + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        return APDModelOutput(hidden_states=hidden_states, past_key_values=presents)

    def initialise_weights(self, config: APDConfig) -> None:
        pretrained_gpt2 = GPT2Model.from_pretrained(
            config.gpt2_hf_model,
            config=self.gpt2_config,
            ignore_mismatched_sizes=True
        )

        # Load positional embeddings
        self.positional_embedding.weight.data[:config.max_position_embeddings,
                                         :] = pretrained_gpt2.wpe.weight.data[:config.max_position_embeddings, :]


        for i, (hidden_layer, pretrained_hidden_layer) in enumerate(zip(self.hidden_layers, pretrained_gpt2.h)):
            hidden_layer.ln_1.load_state_dict(
                pretrained_hidden_layer.ln_1.state_dict())
            hidden_layer.ln_2.load_state_dict(
                pretrained_hidden_layer.ln_2.state_dict())
            hidden_layer.mlp.load_state_dict(
                pretrained_hidden_layer.mlp.state_dict())

            # Load attention weights
            hidden_layer.attn.c_attn.weight.data = pretrained_hidden_layer.attn.c_attn.weight.data.clone()
            hidden_layer.attn.c_attn.bias.data = pretrained_hidden_layer.attn.c_attn.bias.data.clone()
            hidden_layer.attn.c_proj.load_state_dict(
                pretrained_hidden_layer.attn.c_proj.state_dict())

            if hasattr(hidden_layer, 'crossattention'):
                # Initialize cross-attention weights appropriately
                hidden_layer.crossattention.c_attn.weight.data = pretrained_hidden_layer.attn.c_attn.weight.data.clone()
                hidden_layer.crossattention.c_attn.bias.data = pretrained_hidden_layer.attn.c_attn.bias.data.clone()
                hidden_layer.crossattention.c_proj.load_state_dict(
                    pretrained_hidden_layer.attn.c_proj.state_dict())
                hidden_layer.ln_cross_attn.load_state_dict(
                    pretrained_hidden_layer.ln_1.state_dict())

        self.token_embedding.load_state_dict(pretrained_gpt2.wte.state_dict())

       # Initialize other new modules
        if self.use_multi_scale_features:
            self.multi_scale_feature_extractor.apply(self._init_weights)
        if self.use_dynamic_fusion:
            self.feature_fusion.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class APDLMHeadModel(nn.Module):
    def __init__(self, config: APDConfig):
        super().__init__()
        self.config = config
        self.transformer = APDModel(config)
        self.language_model_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        image_size, patch_size = config.image_size, config.patch_size
        self.image_embedding_length = int(
            (image_size[0] / patch_size[0]) * (image_size[1] / patch_size[1]))

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        labels: Optional[torch.LongTensor] = None,
    ) -> APDLMHeadModelOutput:

        transformer_output = self.transformer(
            pixel_values=pixel_values,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        logits = self.language_model_head(transformer_output.hidden_states)

        loss, accuracy = None, None
        if labels is not None:
            labels = labels.to(logits.device)

            # Shift so that tokens < n predict n
            shift_logits = logits[...,
                                  self.image_embedding_length:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            label_matches = shift_labels.view(-1) == torch.argmax(
                torch.nn.functional.softmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1), dim=-1
            )

            # reduce loss
            if attention_mask is not None:
                mask = attention_mask[..., 1:].reshape(-1)

                loss = (mask * loss).sum() / mask.sum()
                accuracy = (mask * label_matches).sum() / mask.sum()
            else:
                loss = loss.mean()
                accuracy = torch.sum(label_matches) / label_matches.shape[0]

        return APDLMHeadModelOutput(
            loss=loss,
            logits=logits,
            accuracy=accuracy,
            past_key_values=transformer_output.past_key_values
        )

    @torch.no_grad()
    def generate(
            self,
            inputs: APDProcessorOutput,
            processor: APDProcessor,
            num_beams: int = 1,
            use_cache: bool = True
    ):
        # params and configs
        batch_size = inputs.input_ids.shape[0]
        model_kwargs = {
            'pixel_values': inputs.pixel_values,
            'attention_mask': inputs.attention_mask,
            'use_cache': use_cache
        }
        generation_config = GenerationConfig(
            max_new_tokens=1,
            pad_token_id=processor.tokeniser.pad_token_id,
            eos_token_id=processor.tokeniser.eos_token_id,
            bos_token_id=processor.tokeniser.bos_token_id,
            num_beams=num_beams,
            max_length=processor.tokeniser.model_max_length
        )

        # interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=inputs.input_ids,
            expand_size=generation_config.num_beams,
            **model_kwargs,
        )

        # prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            processor=processor
        )

        if num_beams > 1:
            # prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs.input_ids.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )

        elif num_beams == 1:
            result = self._sample(
                input_ids,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )
        else:
            raise ValueError("num_beams must be a positive integer.")

        return result

    def _sample(
        self,
        input_ids: torch.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> torch.Tensor:
        # init values
        pad_token_id = generation_config.pad_token_id
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(
            input_ids, model_kwargs)

        this_peer_finished = False
        while not this_peer_finished:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first
            # iteration (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # token selection
            next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + \
                    pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update generated ids, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        return input_ids

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> torch.Tensor:
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(
            input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False
        decoder_prompt_len = input_ids.shape[-1]
        while not this_peer_finished:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first
            # iteration (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(
                input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_tokens_to_keep = max(2, 1 + 1) * num_beams
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(
                next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            decoder_prompt_len=decoder_prompt_len,
        )

        return sequence_outputs["sequences"]

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        processor: Optional[APDProcessor] = None,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(
                self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(
                max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            if processor is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(
                stop_strings=generation_config.stop_strings, tokenizer=processor.tokeniser)
            )
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(
                eos_token_id=generation_config.eos_token_id))
        return criteria

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> tuple[tuple[Tensor, ...], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in past_key_values
        )

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: APDLMHeadModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        # update cache
        model_kwargs['past_key_values'] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        return model_kwargs

    @staticmethod
    def prepare_inputs_for_generation(
        input_ids: torch.Tensor, past_key_values=None, **kwargs
    ) -> Dict[str, Any]:
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "pixel_values": kwargs.get("pixel_values"),
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _get_initial_cache_position(input_ids, model_kwargs):
        if not model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = None
            return model_kwargs

        model_kwargs["cache_position"] = torch.arange(
            0, input_ids.shape[-1], device=input_ids.device)
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: Optional[torch.LongTensor],
        expand_size: int = 1,
        **model_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                        key != "cache_position"
                        and dict_to_expand[key] is not None
                        and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(
                        expand_size, dim=0)
            return dict_to_expand

        input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        return input_ids, model_kwargs
