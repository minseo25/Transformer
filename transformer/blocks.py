import torch
import math
from transformer.abstract import Module
from transformer.layers import Linear, Activation, Softmax, RMSNorm, RotaryPosEmb

class MLP(Module):
    """
    Computes a MLP layer: y = MLP(x) with given hidden dimensions

    Parameters:
        linear_0 (Linear): First linear layer of MLP (dimensions[0] -> dimensions[1])
        act_fn_0 (Activation): First activation function of MLP (dimensions[1] -> dimensions[1])
        linear_1 (Linear): Second linear layer of MLP (dimensions[1] -> dimensions[2])
        act_fn_1 (Activation): Second activation function of MLP (dimensions[2] -> dimensions[2])
        ...
        final (Linear): Last linear layer of MLP (dimensions[-2] -> dimensions[-1])

    Inputs:
        x (torch.Tensor): Input tensor of shape (..., dimensions[0])

    Returns:
        y (torch.Tensor): Output tensor of shape (..., dimensions[-1])
    """
    def __init__(self, *dimensions):
        super().__init__()
        self.dimensions = dimensions

        for i in range(len(dimensions) - 2):
            setattr(self, f"linear_{i}", Linear(dimensions[i], dimensions[i+1]))
            setattr(self, f"act_fn_{i}", Activation("silu"))

        self.final = Linear(dimensions[-2], dimensions[-1])
        self.sub_modules = [f"linear_{i}" for i in range(len(dimensions) - 2)] + ["final"]

    def forward(self, x):
        """
        Compute y = MLP(x) using internal Linear and Activation modules

        Inputs:
            x (torch.Tensor): Input tensor of shape (..., dimensions[0])

        Returns:
            y (torch.Tensor): Output tensor of shape (..., dimensions[-1])
        """
        hidden_x = x
        for i in range(len(self.dimensions) - 2):
            linear_i = getattr(self, f"linear_{i}")
            act_fn_i = getattr(self, f"act_fn_{i}")
            hidden_x = act_fn_i(linear_i(hidden_x))
        
        y = self.final(hidden_x)
        return y

    
    def backward(self, grad_y):
        """
        Compute dL/dx, and also call `backward()` for internal Linear and Activation modules

        Inputs:
            grad_y (torch.Tensor): dL/dy tensor of shape (..., dimensions[-1])

        Outputs:
            grad_x (torch.Tensor): dL/dx tensor of shape (..., dimensions[0])
        """
        grad_x = self.final.backward(grad_y)
        for i in range(len(self.dimensions) - 3, -1, -1):
            linear_i = getattr(self, f"linear_{i}")
            act_fn_i = getattr(self, f"act_fn_{i}")
            grad_x = linear_i.backward(act_fn_i.backward(grad_x))

        return grad_x
    

class FeedForward(Module):
    """
    Feed-forward layer of Transformers: y = MLP(x)

    Parameters:
    x (torch.Tensor): Input tensor of shape (batch_size, ..., dim)

    Returns:
    torch.Tensor: Output tensor of shape (batch_size, ..., dim)
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(dim, hidden_dim)
        self.w3 = Linear(hidden_dim, dim)
        self.act_fn = Activation("silu")
        self.sub_modules = ["w1", "w2", "w3"]

    def forward(self, x):
        """
        Forward pass: Compute feed-forward network
        
        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., dim)
            
        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, ..., dim)
        """
        self.x = x
        self.feature1 = self.w1(x)
        self.feature2 = self.w2(x)
        self.gate = self.act_fn(self.feature2)
        self.feature3 = self.feature1 * self.gate

        return self.w3(self.feature3)

    def backward(self, grad_y):
        """
        Backward pass: Compute gradients
        
        Inputs:
            grad_y (torch.Tensor): Output gradient of shape (batch_size, ..., dim)
            
        Returns:
            grad_x (torch.Tensor): Input gradient of shape (batch_size, ..., dim)
        """
        # (..., dim) => (..., hidden_dim)
        grad_feature3 = self.w3.backward(grad_y)
        # (..., hidden_dim) => (..., hidden_dim)
        grad_gate = self.feature1 * grad_feature3
        grad_feature1 = grad_feature3 * self.gate
        grad_feature2 = self.act_fn.backward(grad_gate)
        # (..., hidden_dim) => (..., dim), copy gate => gradient sum
        grad_x = self.w1.backward(grad_feature1) + self.w2.backward(grad_feature2)

        return grad_x


class Attention(Module):
    """
    Attention layer of Transformers: y = Attention(x)

    Parameters:
    x (torch.Tensor): Input tensor of shape (batch_size, seq, dim)
    mask (torch.Tensor): Input tensor of shape (batch_size, seq, seq)

    Returns:
    torch.Tensor: Output tensor of shape (batch_size, seq, dim)
    """

    def __init__(self, dim, head_dim, num_head):
        self.head_dim = head_dim
        self.num_head = num_head

        self.wq = Linear(dim, head_dim * num_head)
        self.wk = Linear(dim, head_dim * num_head)
        self.wv = Linear(dim, head_dim * num_head)
        self.wo = Linear(head_dim * num_head, dim)

        self.rot_pos_emb = RotaryPosEmb()
        self.softmax = Softmax()
        self.sub_modules = ["wq", "wk", "wv", "wo"]

    def forward(self, x, cos, sin, mask=None, return_attn=False):
        """
        Forward pass: Compute attention mechanism
        
        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            mask (torch.Tensor, optional): Attention mask of shape (seq_len, seq_len)
            return_attn (bool): Whether to return attention scores
            
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, dim)
            attn_scores (torch.Tensor, optional): Attention scores if return_attn=True
        """
        self.x = x
        bsz, seqlen, _ = x.shape
        # (bsz, seqlen, dim) => (bsz, seqlen, head_dim * num_head)
        Q, K, V = self.wq(x), self.wk(x), self.wv(x)
        # (bsz, seqlen, head_dim * num_head) => (bsz, num_head, seqlen, head_dim)
        Q = Q.reshape(bsz, seqlen, self.num_head, self.head_dim).transpose(1,2)
        K = K.reshape(bsz, seqlen, self.num_head, self.head_dim).transpose(1,2)
        self.V = V.reshape(bsz, seqlen, self.num_head, self.head_dim).transpose(1,2)

        # positional encoding
        self.Q, self.K = self.rot_pos_emb(Q, K, cos, sin)

        # (bsz, num_head, seqlen, seqlen)
        scores = torch.matmul(self.Q, self.K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        # apply masking
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
            scores += mask.to(scores.device)
        self.attn_scores = self.softmax(scores)

        # (bsz, num_head, seqlen, seqlen) * (bsz, num_head, seqlen, head_dim) => (bsz, num_head, seqlen, head_dim)
        output = torch.matmul(self.attn_scores, self.V)
        # (bsz, num_head, seqlen, head_dim) => (bsz, seqlen, head_dim * num_head)
        self.output = output.transpose(1,2).reshape(bsz, seqlen, -1)
        # (bsz, seqlen, head_dim * num_head) => (bsz, seqlen, dim)
        self.y = self.wo(self.output)

        if return_attn:
            return self.y, self.attn_scores
        return self.y
    
    def backward(self, grad_y):
        """
        Backward pass: Compute gradients for attention mechanism
        
        Inputs:
            grad_y (torch.Tensor): Gradient of output tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            grad_x (torch.Tensor): Gradient of input tensor of shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = grad_y.shape
        # (bsz, seqlen, dim) => (bsz, num_head, seqlen, head_dim)
        grad_output = self.wo.backward(grad_y).reshape(bsz, seqlen, self.num_head, self.head_dim).transpose(1,2)
        # (..., seqlen, head_dim) * (..., head_dim, seqlen) => (bsz, num_head, seqlen, seqlen)
        grad_attn_scores = torch.matmul(grad_output, self.V.transpose(-2,-1))
        # (..., seqlen, seqlen) * (..., seqlen, head_dim) => (bsz, num_head, seqlen, head_dim)
        grad_value = torch.matmul(self.attn_scores.transpose(-2,-1), grad_output)

        # (bsz, num_head, seqlen, seqlen)
        grad_scores = self.softmax.backward(grad_attn_scores)
        grad_scores *= 1 / math.sqrt(self.head_dim)

        # (..., seqlen, seqlen) * (..., seqlen, head_dim) => (bsz, num_head, seqlen, head_dim)
        grad_query = torch.matmul(grad_scores, self.K)
        # (..., seqlen, seqlen) * (..., seqlen, head_dim) => (bsz, num_head, seqlen, head_dim)
        grad_key = torch.matmul(grad_scores.transpose(-1,-2), self.Q)

        # positional encoding
        grad_query, grad_key, grad_cos, grad_sin = self.rot_pos_emb.backward(grad_query, grad_key)
        
        # (bsz, num_head, seqlen, head_dim) => (bsz, num_head, dim), copy gate => gradient sum
        grad_query = grad_query.transpose(1,2).reshape(bsz, seqlen, -1)
        grad_key = grad_key.transpose(1,2).reshape(bsz, seqlen, -1)
        grad_value = grad_value.transpose(1,2).reshape(bsz, seqlen, -1)
        grad_x = self.wq.backward(grad_query) + self.wk.backward(grad_key) + self.wv.backward(grad_value)

        return grad_x, grad_cos, grad_sin

    
class TransformerBlock(Module):
    """
    Transformer Block: Attention + Feed-Forward Network + Layer Normalization + Residual Connection

    Parameters:
        layer_id (int): Layer ID of the block
        dim (int): Model dimension
        num_head (int): Number of attention heads
    """
    def __init__(self, layer_id, dim, num_head):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.attention = Attention(self.dim, self.head_dim, self.num_head)
        self.ffn = FeedForward(self.dim, self.dim * 4)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()

        self.sub_modules = ["attention", "ffn"]

    def forward(self, x, cos, sin, mask=None):
        """
        A residual block layer of Transformers: y = Block(x)

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq, dim)
            position_embeddings (Tuple of torch.Tensor): cos and sine values for RoPE
            mask (torch.Tensor): Input tensor of shape (batch_size, seq, seq)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq, dim)
        """
        # (bsz, seqlen, dim) => (bsz, seqlen, dim)
        x_norm = self.attention_norm(x)
        h = self.attention(x_norm, cos, sin, mask) + x

        # (bsz, seqlen, dim) => (bsz, seqlen, dim)
        h_norm = self.ffn_norm(h)
        y = self.ffn(h_norm) + h

        return y
    
    def backward(self, grad_y):
        """
        Calculate gradients of each input.
        You should use `backward()` of each submodule, instead of calculating their gradient manually.

        Parameters:
            grad_output (torch.Tensor): A gradient of output of shape (batch_size, seq, dim)

        Returns:
            grad_x (torch.Tensor): A gradient of x of shape (batch_size, seq, dim)
            grad_cos (torch.Tensor): A gradient of cos of shape (batch_size, seq, dim)
            grad_sin (torch.Tensor): A gradient of sin of shape (batch_size, seq, dim)
        """
        # (bsz, seqlen, dim) => (bsz, seqlen, dim)
        grad_h_norm = self.ffn.backward(grad_y)
        grad_h = self.ffn_norm.backward(grad_h_norm) + grad_y

        # (bsz, seqlen, dim) => (bsz, seqlen, dim)
        grad_x_norm, grad_cos, grad_sin = self.attention.backward(grad_h)
        grad_x = self.attention_norm.backward(grad_x_norm) + grad_h

        return grad_x, grad_cos, grad_sin
