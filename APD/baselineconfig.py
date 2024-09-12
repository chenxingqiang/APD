class BaselineConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 256,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_sequence_length: int = 256,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        image_size: tuple = (32, 128),  # (height, width)
        patch_size: tuple = (4, 8),     # (height, width)
        num_channels: int = 3,
        use_multi_scale_features: bool = False,
        use_dynamic_fusion: bool = False,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_multi_scale_features = use_multi_scale_features
        self.use_dynamic_fusion = use_dynamic_fusion

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
