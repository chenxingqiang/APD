from typing import Optional, Union, Tuple, List, Literal


class APDConfig:
    def __init__(
        self,
        # Image configuration
        image_size: Optional[Union[Tuple[int], List[int]]] = (
            32, 128),  # (height, width)
        num_channels: Optional[int] = 3,

        # Training configuration
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        min_learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        warmup_epochs: int = 5,

        # Data augmentation
        aug_prob: float = 0.5,
        rotation_range: int = 10,
        scale_range: float = 0.2,

        # Detection parameters
        min_text_confidence: float = 0.5,
    ):
        # Image configuration
        self.image_size = image_size
        self.num_channels = num_channels

        # Training configuration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        # Data augmentation
        self.aug_prob = aug_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        # Detection parameters
        self.min_text_confidence = min_text_confidence

    def to_dict(self):
        """Converts configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        """Creates a configuration instance from a dictionary."""
        return cls(**config_dict)
