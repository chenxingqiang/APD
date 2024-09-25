from typing import Optional, Union, Tuple, List, Literal
from transformers.models.gpt2.modeling_gpt2 import GPT2Config


class APDConfig:
    def __init__(
        self,
        gpt2_hf_model: str = 'openai-community/gpt2',
        vit_hf_model: str = 'google/vit-base-patch16-224',
        vocab_size: Optional[int] = 50257,
        max_position_embeddings: Optional[int] = 1024,
        hidden_size: Optional[int] = 768,
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 12,
        patch_size: Optional[Union[Tuple[int], List[int]]] = (
            4, 8),  # (height, width)
        image_size: Optional[Union[Tuple[int], List[int]]] = (
            32, 128),  # (height, width)
        num_channels: Optional[int] = 3,
        resid_pdrop: Optional[float] = 0.1,
        embd_pdrop: Optional[float] = 0.1,
        attn_pdrop: Optional[float] = 0.1,
        layer_norm_epsilon: Optional[float] = 1e-5,
        attn_implementation: Literal['sdpa', 'flash_attention_2'] = 'sdpa',
        # Additional custom configuration options
        num_scales: int = 3,
        scale_factors: List[float] = [0.5, 1.0, 2.0],
        fusion_type: Literal['attention', 'concat', 'adaptive'] = 'attention',
        use_dynamic_fusion: bool = True,
        cross_attention_layers: List[int] = [3, 6, 9],
        use_multi_scale_features: bool = True,
    ):
        self.add_cross_attention = True
        # Load the GPT-2 configuration from Hugging Face model
        self.gpt2_config = GPT2Config.from_pretrained(
            gpt2_hf_model, add_cross_attention=self.add_cross_attention)

        # Use the GPT-2 configuration parameters for this model
        self.vocab_size = self.gpt2_config.vocab_size if vocab_size is None else vocab_size
        self.hidden_size = self.gpt2_config.n_embd if hidden_size is None else hidden_size
        self.num_hidden_layers = self.gpt2_config.n_layer if num_hidden_layers is None else num_hidden_layers
        self.num_attention_heads = self.gpt2_config.n_head if num_attention_heads is None else num_attention_heads
        self.max_position_embeddings = self.gpt2_config.n_positions if max_position_embeddings is None else max_position_embeddings
        self.resid_pdrop = self.gpt2_config.resid_pdrop if resid_pdrop is None else resid_pdrop
        self.embd_pdrop = self.gpt2_config.embd_pdrop if embd_pdrop is None else embd_pdrop
        self.attn_pdrop = self.gpt2_config.attn_pdrop if attn_pdrop is None else attn_pdrop
        self.layer_norm_epsilon = self.gpt2_config.layer_norm_epsilon if layer_norm_epsilon is None else layer_norm_epsilon

        # Store original GPT-2 and ViT model references
        self.gpt2_hf_model = gpt2_hf_model
        self.vit_hf_model = vit_hf_model

        # Image and patch configuration
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels

        # Attention implementation (custom addition)
        self._attn_implementation = attn_implementation

        # GPT-2 related options for advanced control
        self.n_inner = None
        self.scale_attn_weights = True
        self.scale_attn_by_inverse_layer_idx = False
        self.reorder_and_upcast_attn = False
        self.add_cross_attention = False
        self.activation_function = "gelu_new"

        # New multi-scale and fusion-related configuration
        self.num_scales = num_scales
        self.scale_factors = scale_factors
        self.fusion_type = fusion_type
        self.use_dynamic_fusion = use_dynamic_fusion
        self.cross_attention_layers = cross_attention_layers
        self.use_multi_scale_features = use_multi_scale_features

    def to_dict(self):
        """Converts configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        """Creates a configuration instance from a dictionary."""
        return cls(**config_dict)
