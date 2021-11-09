from .idea_test import NaiveClassifier
from .co_attn_pooling import CoAttnWithAttnPoolingClassifier
from .co_attn_pooling_simple import CoAttnWithAttnPoolingSimpleClassifier
from .co_attn_vlad import CoAttnWithNeXtVLADClassifier
from .tvn_attn_pooling import TVNCoAttnPoolingClassifier
from .tvn_base_classifier import TVNBaseClassifier
from .co_attn_pooling_mcg import CoAttnWithAttnPoolingWithMCGClassifier
from .mixer_attn_pooling import MixerWithAttnPoolingClassifier
from .mlp_mixer import MixerClassifier
from .residual_mlp import ResidualMLPClassifier
from .mlp_multi_classifier import MLPMultiClassifier
from .transformer_base import TransformerClassifier
from .transformer_pooling import TransformerPoolingClassifier
from .co_attn_trans_pooling import CoAttnWithTransPoolingClassifier
from .single_transformer_pooling import SingleTransPoolingClassifier
from .co_attn import CoAttnClassifier
from .label_gcn import GCNClassifier
from .mml4torch import VideoTransformer
from .final import FinalClassifier

module_dict = {
    "idea_test": NaiveClassifier,
    "co_attn_pooling": CoAttnWithAttnPoolingClassifier,
    "co_attn_pooling_simple": CoAttnWithAttnPoolingSimpleClassifier,
    "co_attn_vlad": CoAttnWithNeXtVLADClassifier,
    "tvn_attn_pooling": TVNCoAttnPoolingClassifier,
    "tvn_base_classifier": TVNBaseClassifier,
    "co_attn_pooling_mcg": CoAttnWithAttnPoolingWithMCGClassifier,
    "mixer_attn_pooling": MixerWithAttnPoolingClassifier,
    "mlp_mixer": MixerClassifier,
    "residual_mlp": ResidualMLPClassifier,
    "mlp_multi_classifier": MLPMultiClassifier,
    "transformer_base": TransformerClassifier,
    "transformer_pooling": TransformerPoolingClassifier,
    "co_attn_trans_pooling": CoAttnWithTransPoolingClassifier,
    "single_transformer_pooling": SingleTransPoolingClassifier,
    "co_attn": CoAttnClassifier,
    "label_gcn": GCNClassifier,
    "mml4torch": VideoTransformer,
    "final": FinalClassifier,
}
