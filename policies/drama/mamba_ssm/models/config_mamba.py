from dataclasses import dataclass, field


@dataclass
class MambaConfig:

    model_size: str = "D"
    action_dim: int = 5
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    pff_cfg: dict = field(default_factory=dict)
    dropout_p: float = 0.0
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
