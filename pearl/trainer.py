# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# Adapted from LLM-JEPA (https://github.com/rbalestr-lab/llm-jepa), Apache-2.0.
# Modified: multimodal inputs (pixel values / image grids per view), NextLat
# loss with a dynamics head, explicit JEPA distance selection.
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from trl.trainer.sft_trainer import SFTTrainer

JEPA_LOSSES = ("cosine", "smoothl1", "l2", "mse")


class PEARLTrainer(SFTTrainer):
    """SFT trainer with the PEARL objective.

    ``L = L_VLM + lbd * (L_JEPA + L_NextLat)``, where

    * ``L_VLM`` is next-token prediction on the full conversation;
    * ``L_JEPA`` aligns the last-token embedding of the input view (with
      predictor tokens) to the stop-gradient last-token embedding of the
      trajectory view;
    * ``L_NextLat`` trains a dynamics head to predict the next hidden state
      (SmoothL1 to the stop-gradient target) and decodes the prediction through
      a frozen copy of the LM head (cross-entropy to the next token).
    """

    def __init__(self, *args, lbd=0.2, last_token=-2, jepa_loss="cosine", use_jepa=True,
                 nextlat=True, **kwargs):
        assert jepa_loss in JEPA_LOSSES, f"jepa_loss must be one of {JEPA_LOSSES}"
        self.lbd = lbd
        self.last_token = last_token
        self.jepa_loss = jepa_loss
        self.use_jepa = use_jepa
        self.nextlat = nextlat
        super().__init__(*args, **kwargs)

    def _last_token_index(self, input_ids, attention_mask):
        """Index of the embedding token: end of the unpadded sequence + ``last_token``."""
        index = []
        for ids, mask in zip(input_ids, attention_mask):
            length, started = 0, False
            for m in mask:
                if m != 0:
                    started = True
                if m == 0 and started:
                    break
                length += 1
            index.append(length + self.last_token)
        return torch.tensor(index, device=input_ids.device)

    def _forward(self, model, inputs):
        """One forward pass over [full ; input view ; trajectory view]."""
        views = ("", "_user", "_assistant")
        llm_inputs = {
            key: torch.cat([inputs[f"{key}{v}"] for v in views], dim=0)
            for key in ("input_ids", "labels", "attention_mask")
        }
        pixel_values = [inputs[f"pixel_values{v}"] for v in views if inputs.get(f"pixel_values{v}") is not None]
        image_grid_thw = [inputs[f"image_grid_thw{v}"] for v in views if inputs.get(f"image_grid_thw{v}") is not None]
        assert pixel_values and image_grid_thw, "No images in batch"
        llm_inputs["pixel_values"] = torch.cat(pixel_values, dim=0)
        llm_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        outputs = model(**llm_inputs, output_hidden_states=True)
        batch_size = inputs["input_ids"].shape[0]
        hidden = outputs.hidden_states[-1]
        input_embeds = model.base_model.get_input_embeddings()(llm_inputs["input_ids"][:batch_size])
        return outputs, hidden, input_embeds, llm_inputs["labels"][:batch_size]

    def _jepa_distance(self, pred, target):
        target = target.detach()
        if self.jepa_loss == "cosine":
            return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        if self.jepa_loss == "smoothl1":
            return SmoothL1Loss()(pred, target)
        if self.jepa_loss == "l2":
            return torch.linalg.norm(pred - target, ord=2, dim=-1).mean()
        return torch.mean((pred - target) ** 2)

    def _nextlat_loss(self, model, states, input_embeds, labels):
        """NextLat with a one-step horizon; returns (state loss, token loss)."""
        batch_size = states.shape[0]
        lm_head = deepcopy(model.base_model.lm_head)
        shift_labels = labels[..., 1:].contiguous().view(-1)

        pred_next = model.base_model.dynamics_head(torch.cat([states[:, :-1, :], input_embeds[:, 1:, :]], dim=-1))
        state_loss = SmoothL1Loss(reduction="none")(pred_next, states.detach()[:, 1:, :])
        mask = (shift_labels.reshape(batch_size, -1) > -100).unsqueeze(2)
        state_loss = state_loss * mask
        state_loss = (state_loss.sum(dim=(-2, -1)) / (mask.sum(dim=-2).squeeze(-1) * state_loss.shape[-1])).sum() / batch_size

        pred_logits = lm_head(pred_next).contiguous()
        pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
        token_loss = CrossEntropyLoss()(pred_logits, shift_labels.to(pred_logits.device))
        return state_loss, token_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch_size = inputs["input_ids_user"].shape[0]
        index_user = self._last_token_index(inputs["input_ids_user"], inputs["attention_mask_user"])
        index_traj = self._last_token_index(inputs["input_ids_assistant"], inputs["attention_mask_assistant"])

        outputs, hidden, input_embeds, labels = self._forward(model, inputs)
        vlm_loss = outputs.loss
        total_loss = vlm_loss

        jepa_loss = torch.zeros((), device=vlm_loss.device)
        if self.use_jepa:
            rows = range(batch_size)
            pred = hidden[batch_size:2 * batch_size][rows, index_user, :]
            target = hidden[2 * batch_size:][rows, index_traj, :]
            jepa_loss = self._jepa_distance(pred, target)
            total_loss = total_loss + self.lbd * jepa_loss

        state_loss = token_loss = torch.zeros((), device=vlm_loss.device)
        if self.nextlat:
            state_loss, token_loss = self._nextlat_loss(model, hidden[:batch_size], input_embeds, labels)
            total_loss = total_loss + self.lbd * (state_loss + token_loss)

        self.log({
            "loss_total": total_loss.detach().item(),
            "loss_vlm": vlm_loss.detach().item(),
            "loss_jepa": jepa_loss.detach().item(),
            "loss_nextlat_state": state_loss.detach().item(),
            "loss_nextlat_token": token_loss.detach().item(),
        })
        return (total_loss, outputs) if return_outputs else total_loss
