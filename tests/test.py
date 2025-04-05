import torch
from transformer.layers import Linear, Activation, Embedding, Softmax, RMSNorm, CELoss, RotaryEmbedding
from transformer.blocks import MLP, FeedForward, Attention, TransformerBlock
from transformer.model import Transformer
from utils.util import test_module

# Layers
test_module("Linear 2D input", Linear, [6, 10], torch.randn(5, 6))
test_module("Linear 3D input", Linear, [6, 10], torch.randn(5, 3, 6))
test_module("Linear 4D input", Linear, [3, 4], torch.randn(5, 2, 4, 3))

test_module("Activation", Activation, ["silu"], torch.randn(5, 3))

test_module("Embedding", Embedding, [3, 4], torch.randint(0, 3, (5, 3)))

test_module("Softmax", Softmax, [], torch.randn(3, 4))

# Blocks
test_module("MLP", MLP, [3, 4, 5, 2], torch.randn(5, 3))
test_module("MLP", MLP, [3, 4, 5, 6, 2], torch.randn(5, 3))

test_module("FeedForward", FeedForward, [3, 4], torch.randn(5, 3))

batch_size, seq_len, vocab_size = 5, 4, 7
n_layers, n_head, head_dim = 2, 3, 4
dim = n_head * head_dim
# random embedding
h = torch.randn(batch_size, seq_len, dim)
# generate rope with random position_ids
cos, sin = RotaryEmbedding(head_dim)(torch.randint(0, seq_len, (batch_size, seq_len)))

test_module("Attention", Attention, [dim, head_dim, n_head], [h, cos, sin])
test_module("TransformerBlock", TransformerBlock, [0, dim, n_head], [h, cos, sin])

tok_ids = torch.randint(0, vocab_size, (batch_size, seq_len)) # random tokenizing
test_module("Transformer", Transformer, [n_layers, vocab_size, dim, n_head], [tok_ids, tok_ids])