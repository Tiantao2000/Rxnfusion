from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .configuration_rxnlm import RxnLMConfig
from .esmc import (
    ESMCForMaskedLM,
    ESMCForSequenceClassification,
    ESMCModel,
    ESMCOutput,
    Pooler,
    RegressionHead,
)


class RxnLMForMaskedLM(ESMCForMaskedLM):
    base_model_prefix = "rxnlm"
    supports_gradient_checkpointing = True
    config_class = RxnLMConfig


class RxnLMForSequenceClassification(ESMCForSequenceClassification):
    base_model_prefix = "rxnlm"
    supports_gradient_checkpointing = True
    config_class = RxnLMConfig


class RxnLMForHierarchicalClassification(ESMCModel):
    base_model_prefix = "rxnlm"
    supports_gradient_checkpointing = True
    config_class = RxnLMConfig

    def __init__(
        self,
        config: RxnLMConfig,
        label_sizes: dict[str, int],
        loss_weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.label_sizes = label_sizes
        self.loss_weights = loss_weights or {name: 1.0 for name in label_sizes}
        self.pooler = Pooler(["cls", "mean"])
        self.classifiers = torch.nn.ModuleDict(
            {
                name: RegressionHead(
                    config.hidden_size * 2,
                    num_labels,
                    config.hidden_size * 4,
                )
                for name, num_labels in label_sizes.items()
            }
        )
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> ESMCOutput:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        features = self.pooler(output.last_hidden_state, attention_mask)
        logits = {
            name: classifier(features)
            for name, classifier in self.classifiers.items()
        }
        loss = None
        if labels is not None:
            labels = labels.to(features.device).long()
            losses = []
            for index, name in enumerate(self.label_sizes):
                weight = self.loss_weights.get(name, 1.0)
                losses.append(weight * F.cross_entropy(logits[name], labels[:, index]))
            loss = sum(losses)
        return ESMCOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
