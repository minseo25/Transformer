import torch
from transformer.abstract import Module
from transformer.layers import Embedding, RotaryEmbedding, RMSNorm, Linear, CELoss
from transformer.blocks import TransformerBlock

class Transformer(Module):
    """
    Transformer model implementation with attention mechanism and positional embeddings.
    """
    def __init__(self, n_layers, vocab_size, dim, num_head):
        super().__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_head = num_head
        self.head_dim = dim // num_head

        self.tok_embeddings = Embedding(vocab_size, dim)
        for layer_id in range(n_layers):
            setattr(self, f'layer_{layer_id}', TransformerBlock(layer_id, dim, num_head))
        
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm = RMSNorm()
        self.output = Linear(dim, vocab_size)
        self.cross_entropy_loss = CELoss()

        self.sub_modules = ["tok_embeddings"] + [f"layer_{i}" for i in range(n_layers)] + ["output"]

    def forward(self, tokens, labels=None):
        """
        Transformer architecture: y = Transformer(x, label=None)

        Parameters:
            tokens (torch.Tensor): Input tokens of shape (batch_size, seq)
            labels (torch.Tensor): Input tokens of shape (batch_size, seq)

        Returns:
            loss (torch.Tensor): Scalar loss value.
            logits (torch.Tensor): Logits for the next token prediction. shape (batch_size, seq, vocab_size)
        """
        seqlen = tokens.shape[1]

        # (bsz, seqlen) => (bsz, seqlen, dim)
        h = self.tok_embeddings(tokens)

        # random initialize of pos_ids (can be trained), move to device
        position_ids = torch.arange(seqlen, device=tokens.device).reshape(1,-1)
        # set position_embedding
        cos, sin = self.rope(position_ids)

        # define causal mask for attention, move to device
        mask = None
        if seqlen > 1:
            mask = torch.triu(torch.ones(seqlen, seqlen, device=tokens.device), diagonal = 1)
            mask = mask.masked_fill(mask == 1, float("-inf"))

        # (bsz, seqlen, dim)
        for i in range(self.n_layers):
            layer = getattr(self, f"layer_{i}")
            h = layer(h, cos, sin, mask)
        h_norm = self.norm(h)

        # (bsz, seqlen, dim) => (bsz, seqlen, vocab_size)
        logits = self.output(h_norm)

        if labels is not None:
            # (bsz, seqlen, vocab_size) => (1,)
            loss = self.cross_entropy_loss(logits, labels)
            return logits, loss
        return logits

    def backward(self, grad_logits, grad_loss=None):
        """
        Backward pass for gradient computation.

        Args:
            grad_logits: Gradient of logits
            grad_loss: Optional gradient of loss

        Returns:
            grad_input: Gradient of input tokens
            grad_labels: Gradient of labels (if loss was computed)
        """
        grad_labels = None
        
        # Handle loss gradient if provided
        if grad_loss is not None:
            # (1,) => (bsz, seqlen, vocab_size)
            _grad_logits, grad_labels = self.cross_entropy_loss.backward(grad_loss)
            grad_logits += _grad_logits
        
        # (bsz, seqlen, vocab_size) => (bsz, seqlen, dim)
        grad_h_norm = self.output.backward(grad_logits)
        grad_h = self.norm.backward(grad_h_norm)

        for i in range(self.n_layers-1,-1,-1):
            layer = getattr(self, f"layer_{i}")
            grad_h, _, _ = layer.backward(grad_h)

        # (bsz, seqlen, dim) => (bsz, seqlen)
        grad_input = self.tok_embeddings.backward(grad_h)

        return grad_input, grad_labels
