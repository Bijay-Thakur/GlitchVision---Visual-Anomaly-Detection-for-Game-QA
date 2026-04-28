"""Feature extraction (pluggable visual backbones).

The default backbone is ImageNet-pretrained ResNet-18. Optional backbones
(``dino``, ``clip``) are lazily loaded and only used when explicitly
requested and their dependencies are available.
"""

from .embedding_extractor import (  # noqa: F401
    EmbeddingExtractor,
    available_backbones,
)
from .temporal_features import (  # noqa: F401
    TemporalFeatureResult,
    compute_temporal_features,
    temporal_feature_schema,
)
