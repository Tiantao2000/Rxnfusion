from .configuration_esmc import ESMCConfig


class RxnLMConfig(ESMCConfig):
    model_type = "rxnlm"

    def __init__(
        self,
        vocab_size: int = 2362,  # molformer vocab size 236
        hidden_size: int = 1152,
        num_attention_heads: int = 18,
        num_hidden_layers: int = 36,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            **kwargs,
        )
