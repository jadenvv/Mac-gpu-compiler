"""Model hyperparameters for Qwen2-MoE (Qwen1.5-MoE-A2.7B)."""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen -> hashable -> usable as a jit static arg
class Config:
    vocab_size: int
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_experts: int
    top_k: int
    moe_inter: int  # per-expert FFN width
    shared_inter: int  # shared-expert FFN width
    rope_theta: float
    eps: float
    norm_topk: bool  # renormalize the top-k routing weights to sum to 1

    @classmethod
    def from_hf(cls, d: dict) -> "Config":
        assert d["model_type"] == "qwen2_moe", d["model_type"]
        assert d.get("decoder_sparse_step", 1) == 1 and not d.get("mlp_only_layers")
        return cls(
            vocab_size=d["vocab_size"],
            hidden=d["hidden_size"],
            n_layers=d["num_hidden_layers"],
            n_heads=d["num_attention_heads"],
            n_kv_heads=d["num_key_value_heads"],
            head_dim=d["hidden_size"] // d["num_attention_heads"],
            n_experts=d["num_experts"],
            top_k=d["num_experts_per_tok"],
            moe_inter=d["moe_intermediate_size"],
            shared_inter=d["shared_expert_intermediate_size"],
            rope_theta=d["rope_theta"],
            eps=d["rms_norm_eps"],
            norm_topk=d["norm_topk_prob"],
        )

    @classmethod
    def tiny(cls) -> "Config":
        """Same structure, toy sizes: for CPU tests with random weights."""
        return cls(
            vocab_size=512, hidden=128, n_layers=2, n_heads=4, n_kv_heads=4,
            head_dim=32, n_experts=8, top_k=2, moe_inter=64, shared_inter=128,
            rope_theta=1e6, eps=1e-6, norm_topk=False,
        )
