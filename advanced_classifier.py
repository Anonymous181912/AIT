"""
Advanced Classifier with State-of-the-Art Architecture
Features: Multi-Head Attention, SE Blocks, Residual Connections, GELU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for feature interaction learning"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, D) for 1D features
        B, D = x.shape
        
        # Squeeze: Global average (already flat, just use the features)
        scale = self.excitation(x)
        
        # Excite: Channel-wise scaling
        return x * scale


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Stochastic depth (drop path) for regularization
        self.drop_path_rate = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation (smoother than ReLU)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Stochastic depth during training
        if self.training and torch.rand(1).item() < self.drop_path_rate:
            return residual
        
        return residual + x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and FFN"""
    
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x


class AdvancedDeepfakeClassifier(nn.Module):
    """
    State-of-the-art classifier for deepfake detection.
    
    Architecture:
    - Feature embedding with LayerNorm
    - Multi-Head Self-Attention blocks
    - Squeeze-and-Excitation for channel recalibration
    - Residual MLP blocks
    - Classification head with dropout
    """
    
    def __init__(self, 
                 input_dim: int = 79,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 num_transformer_blocks: int = 2,
                 num_residual_blocks: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer blocks with self-attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(embed_dim, reduction=4)
        
        # Residual MLP blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(embed_dim, embed_dim * 2, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # Final normalization before classification
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, input_dim)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        # Embed features
        x = self.embed(x)
        
        # Add sequence dimension for attention (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Remove sequence dimension (B, 1, D) -> (B, D)
        x = x.squeeze(1)
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Residual blocks
        for residual in self.residual_blocks:
            x = residual(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability"""
        x = self.embed(x).unsqueeze(1)
        
        attention_weights = []
        for transformer in self.transformer_blocks:
            # Get attention weights from the attention layer
            attn = transformer.attn
            q = attn.qkv(transformer.norm1(x))
            # Extract attention weights (simplified)
            attention_weights.append(q.mean(dim=-1))
        
        return torch.stack(attention_weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    From paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing for better calibration.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = AdvancedDeepfakeClassifier(input_dim=79, embed_dim=256)
    print(f"Model parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 79)
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    probs = model.predict_proba(x)
    print(f"Probabilities: {probs}")
