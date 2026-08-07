import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dropout_rate=0.0):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)

    def forward(self, x, mask=None):
        attn_logits = self.attention(x).squeeze(-1)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(
                ~mask,
                torch.finfo(attn_logits.dtype).min
            )

        weights = torch.softmax(attn_logits, dim=1)

        if self.training and self.attn_dropout.p > 0:
            weights = self.attn_dropout(weights)
            weights = weights / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)

        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return pooled, weights


class OriginalUNKAI(nn.Module):
    def __init__(self, embedding_dim=2560):
        super().__init__()

        self.pool = AttentionPooling(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1599),
            nn.BatchNorm1d(1599),
            nn.ReLU(),
            nn.Dropout(0.302766),
            nn.Linear(1599, 781),
            nn.BatchNorm1d(781),
            nn.ReLU(),
            nn.Dropout(0.302766),
            nn.Linear(781, 117),
            nn.ReLU(),
            nn.Linear(117, 1),
            nn.Sigmoid(),
        )

    def forward(self, mat1, mat2, mask1=None, mask2=None):
        vec1, _ = self.pool(mat1, mask1)
        vec2, _ = self.pool(mat2, mask2)
        diff = torch.abs(vec1 - vec2)
        prob = self.classifier(diff).squeeze(-1)
        return prob


class SeenUnseenUNKAI(nn.Module):
    def __init__(self, embedding_dim=2560):
        super().__init__()

        self.pool = AttentionPooling(
            embedding_dim,
            attention_dropout_rate=0.16272949251356297,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1408),
            nn.BatchNorm1d(1408),
            nn.ReLU(),
            nn.Dropout(0.5282298379490304),
            nn.Linear(1408, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.5282298379490304),
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, mat1, mat2, mask1=None, mask2=None):
        vec1, _ = self.pool(mat1, mask1)
        vec2, _ = self.pool(mat2, mask2)
        diff = torch.abs(vec1 - vec2)
        logits = self.classifier(diff).squeeze(-1)
        return logits
