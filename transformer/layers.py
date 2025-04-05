import torch
import math
from transformer.abstract import Module

class Linear(Module):
    """
    Computes a linear function: y = xW + b

    Parameters:
        weight (torch.Tensor): Weight tensor of shape (input_dim, output_dim)
        bias (torch.Tensor): Bias tensor of shape (output_dim)

    Inputs:
        x (torch.Tensor): Input tensor of shape (..., input_dim)

    Returns:
        y (torch.Tensor): Output tensor of shape (..., output_dim)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.randn(input_dim, output_dim) / math.sqrt(input_dim) # for consistent variance
        self.bias = torch.zeros(output_dim)
        self.sub_modules = ["weight", "bias"]

    def forward(self, x):
        """
        Compute y = xW + b, and also save input `x` internally for future `backward()` call

        Inputs:
            x (torch.Tensor): Input tensor of shape (..., input_dim)

        Returns:
            y (torch.Tensor): Output tensor of shape (..., output_dim)
        """
        # (..., input_dim) => (..., output_dim)
        y = torch.matmul(x, self.weight) + self.bias
        self.x = x
        return y
    
    def backward(self, grad_y):
        """
        Compute dL/dx, and also save dL/dW and dL/db internally

        Inputs:
            grad_y (torch.Tensor): dL/dy tensor of shape (..., output_dim)

        Outputs:
            grad_x (torch.Tensor): dL/dx tensor of shape (..., input_dim)
        """
        flatten_grad_y = grad_y.reshape(-1, self.output_dim)
        flatten_x = self.x.reshape(-1, self.input_dim)
        # (input_dim, ...) * (..., output_dim) => (input_dim, output_dim)
        self.grad_weight = torch.matmul(flatten_x.T, flatten_grad_y)
        # (..., output_dim) => (output_dim,)
        self.grad_bias = torch.sum(flatten_grad_y, dim=0)
        # (..., output_dim) * (output_dim, input_dim) => (..., input_dim)
        grad_x = torch.matmul(grad_y, self.weight.T)

        return grad_x


class Activation(Module):
    """
    Computes a activation function: y = act(x)

    Parameters:
        name (str): Name of activation function ("sigmoid", "tanh", "relu", "silu")

    Inputs:
        x (torch.Tensor): Input tensor of shape (..., dim)

    Returns:
        y (torch.Tensor): Output tensor of shape (..., dim)
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        """
        Compute activation function and save input and output for backward pass

        Inputs:
            x (torch.Tensor): Input tensor of shape (..., dim)

        Returns:
            y (torch.Tensor): Output tensor of shape (..., dim)
        """
        self.x = x
        if self.name == "sigmoid":
            self.y = 1 / (1 + torch.exp(-x))
        elif self.name == "tanh":
            self.y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        elif self.name == "relu":
            self.y = torch.clamp(x, min=0.)
        elif self.name == "silu":
            self.y = x / (1 + torch.exp(-x))
        else:
            raise ValueError("Unsupported activation function")
        
        return self.y
        
    def backward(self, grad_y):
        """
        Compute gradient of activation function

        Inputs:
            grad_y (torch.Tensor): Gradient of output tensor of shape (..., dim)

        Outputs:
            grad_x (torch.Tensor): Gradient of input tensor of shape (..., dim)
        """
        x = self.x
        y = self.y
        if self.name == "sigmoid":
            grad_x = grad_y * (y * (1 - y))
        elif self.name == "tanh":
            expx = torch.exp(x); expmx = torch.exp(-x)
            grad_x = grad_y * ((expx - expmx) / (expx + expmx))
        elif self.name == "relu":
            grad_x = grad_y * (x > 0.).float()
        elif self.name == "silu":
            expmx = torch.exp(-x)
            grad_x = grad_y * ((1 + expmx + x * expmx) / (1 + expmx) ** 2)
        else:
            raise ValueError("Unsupported activation function")
        
        return grad_x


class Embedding(Module):
    """
    Computes embedding lookup: y = weight[x]

    Parameters:
        weight (torch.Tensor): Embedding weight tensor of shape (vocab_size, dim)

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices

    Returns:
        y (torch.Tensor): Output tensor of shape (batch_size, seq_len, dim)
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = torch.randn(vocab_size, dim) / math.sqrt(dim) # for consistent variance
        self.sub_modules = ["weight"]
    
    def forward(self, x):
        """
        Compute embedding lookup and save one_hot encoding for backward pass

        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, dim)
        """
        # (bsz, seqlen) => (bsz, seqlen, vocab_size)
        self.one_hot = torch.nn.functional.one_hot(x, num_classes=self.vocab_size)
        # (bsz, seqlen, vocab_size) => (bsz, seqlen, dim)
        return self.weight[x]

    def backward(self, grad_y):
        """
        Compute gradient of embedding weights

        Inputs:
            grad_y (torch.Tensor): Gradient of output tensor of shape (batch_size, seq_len, dim)

        Outputs:
            grad_x (torch.Tensor): Zero tensor of shape (batch_size, seq_len) since input is discrete indices
        """
        # (bsz, seqlen, 1, dim) * (bsz, seqlen, vocab_size, 1) => (bsz, seqlen, vocab_size, dim)
        gradient = grad_y.unsqueeze(-2) * self.one_hot.unsqueeze(-1)
        # (bsz, seqlen, vocab_size, dim) => (vocab_size, dim), sum all gradients of single weight value
        self.grad_weight = torch.sum(torch.sum(gradient, dim=0), dim=0)

        # don't care grad_x
        # because it gets discrete index x, one_hot encoding, and find embedding in table
        bsz, seqlen, _ = grad_y.shape
        return torch.zeros(bsz, seqlen)


class Softmax(Module):
    """
    Computes softmax function: y = softmax(x) = exp(x) / sum(exp(x))

    Parameters:
        None

    Inputs:
        x (torch.Tensor): Input tensor of shape (..., seq_len)

    Returns:
        y (torch.Tensor): Output tensor of shape (..., seq_len) where each row sums to 1
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Compute softmax with numerical stability by subtracting max value

        Inputs:
            x (torch.Tensor): Input tensor of shape (..., seq_len)

        Returns:
            y (torch.Tensor): Output tensor of shape (..., seq_len)
        """
        # (..., seqlen)
        x_max, _ = torch.max(x, dim=-1, keepdim=True) # get max along last dimension
        expx = torch.exp(x - x_max) # subtract for numerical stability
        expx_sum = torch.sum(expx, dim=-1, keepdim=True) # sum along last dimension
        self.output = expx / expx_sum

        return self.output

    def backward(self, grad_y):
        """
        Compute gradient of softmax using Jacobian matrix

        Inputs:
            grad_y (torch.Tensor): Gradient of output tensor of shape (..., seq_len)

        Outputs:
            grad_x (torch.Tensor): Gradient of input tensor of shape (..., seq_len)

        Note:
            Jacobian matrix for softmax [s(0) ... s(k)]:
            [s(0)(1-s(0)) s(0)(0-s(1)) ... s(0)(0-s(K))]
            [s(1)(0-s(0)) s(1)(1-s(1)) ... s(1)(0-s(K))]
            ...
            [s(K)(0-s(0)) s(K)(0-s(1)) ... s(K)(1-s(K))]
        """
        seqlen = grad_y.shape[-1]
        repeat_num = [1] * len(grad_y.shape)
        repeat_num[-1] = seqlen # repeat last dim for seqlen times, then reshape to (seqlen, seqlen)

        # (..., seqlen, seqlen)
        diag_matrix = torch.diag_embed(torch.ones_like(self.output))
        grad_repeated = self.output.repeat(repeat_num).reshape(*self.output.shape, seqlen)
        jacobian_matrix = grad_repeated.transpose(-2, -1) * (diag_matrix - grad_repeated)

        # (..., 1, seqlen) * (..., seqlen, seqlen) => (..., 1, seqlen) => (..., seqlen)
        grad_x = torch.matmul(grad_y.unsqueeze(-2), jacobian_matrix).squeeze(-2)
        return grad_x


class RMSNorm(Module):
    """
    Root Mean Square Normalization (RMSNorm)

    When stacking a large number of layers, the distribution of neural network features
    undergoes various transformations and can become significantly different in the final layer.
    In this case, feature values or their gradients may either grow excessively large or shrink drastically.
    Introducing normalization layers at the beginning or end of each layer can help stabilize the computational scale
    and prevents the vanishing/exploding gradient problem.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        # (bsz, seq, dim) => (bsz, seq, 1)
        mean_square = torch.mean(x.pow(2), dim=-1, keepdim=True)
        self.rms = torch.sqrt(mean_square + self.eps)
        return x / self.rms
    
    def forward(self, x):
        """
        Normalization layer of Transformers: y = norm(x)

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq, dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq, dim)
        """
        self.y = self._norm(x)
        return self.y
    
    def backward(self, grad_y):
        """
        Compute gradient of RMSNorm layer.
        
        Parameters:
            grad_y (torch.Tensor): Gradient of output tensor of shape (batch_size, seq, dim)
            
        Returns:
            grad_x (torch.Tensor): Gradient of input tensor of shape (batch_size, seq, dim)
        """
        _, _, dim = grad_y.shape
        # (bsz, seqlen, 1) => (bsz, seqlen, dim, dim)
        diag_term = torch.diag_embed(1 / self.rms.repeat(1, 1, dim))
        # broadcast rms value to last (dim, dim) dimensions
        # (bsz, seqlen, dim, 1) * (bsz, seqlen, 1, dim) => (bsz, seqlen, dim, dim)
        joint_term = (1 / dim) * (1 / self.rms.unsqueeze(-1)) * torch.matmul(self.y.unsqueeze(-1), self.y.unsqueeze(-2))
        jacobian_matrix = diag_term - joint_term
        # (bsz, seqlen, dim, dim) * (bsz, seqlen, dim, 1) => (bsz, seqlen, dim)
        grad_x = torch.matmul(jacobian_matrix, grad_y.unsqueeze(-1)).squeeze(-1)

        return grad_x


class RotaryEmbedding(Module):
    """
    This class computes cosine and sine values for rotation matrices in RoPE.
    """
    def __init__(self, head_dim, base=10000):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        self.theta = self._compute_theta()

    def _compute_theta(self):
        """
        Computes theta values according to the RoPE implementation.
        Generates exponentially decreasing values to have different frequencies for each dimension.

        Returns:
            torch.Tensor: Theta values for RoPE embeddings of shape (head_dim//2,)
        """
        exponent = torch.arange(0, self.head_dim, 2).float() / self.head_dim
        theta = 1 / self.base ** exponent
        return theta
    
    def forward(self, position_ids):
        """
        Computes sinusoidal entries (cosine and sine) for the given position indices.

        Parameters:
            position_ids (torch.Tensor): Position index tensor of shape (batch, seq)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine values of shape (batch, seq, head_dim)
        """
        bsz = position_ids.shape[0]

        # (batch, seq, 1) * (batch, 1, head_dim//2) => (batch, seq, head_dim//2)
        pos_ids = position_ids.unsqueeze(-1).to(dtype=self.theta.dtype)
        theta = self.theta.unsqueeze(0).unsqueeze(0).repeat([bsz, 1, 1]).to(position_ids.device)
        freqs = torch.matmul(pos_ids, theta)

        # (batch, seq, head_dim//2) => (batch, seq, head_dim)
        self.emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = torch.cos(self.emb), torch.sin(self.emb)

        return cos, sin
    
    def backward(self, grad_cos, grad_sin):
        """
        Computes the backward pass for RoPE.

        Parameters:
            grad_cos (torch.Tensor): Gradient of cosine values
            grad_sin (torch.Tensor): Gradient of sine values

        Returns:
            torch.Tensor: Gradient of position indices
        """
        bsz, _, head_dim = grad_cos.shape
        # (batch, seq, head_dim) => (batch, seq, head_dim)
        grad_emb = - torch.sin(self.emb) * grad_cos + torch.cos(self.emb) * grad_sin
        # (batch, seq, head_dim) => (batch, seq, head_dim//2)
        grad_freqs = (grad_emb[:,:,:head_dim//2] + grad_emb[:,:,head_dim//2:]).to(dtype=self.theta.dtype)

        # (batch, seq, head_dim//2) * (batch, head_dim//2, 1) => (batch, seq)
        theta = self.theta.unsqueeze(0).unsqueeze(0).repeat([bsz, 1, 1])
        grad_pos_ids = torch.matmul(grad_freqs, theta.transpose(1,2)).squeeze(-1)
        return grad_pos_ids


class RotaryPosEmb(Module):
    """
    Applies Rotary Position Embedding (RoPE) to query and key vectors.
    
    This class implements the rotation transformation that injects positional information
    into query and key vectors, effectively encoding relative position information.
    """
    def __init__(self):
        super().__init__()

    def forward(self, query, key, cos, sin):
        """
        Applies RoPE to query and key vectors.

        Parameters:
            query (torch.Tensor): Query vectors of shape (batch, num_head, seq, head_dim)
            key (torch.Tensor): Key vectors of shape (batch, num_head, seq, head_dim)
            cos (torch.Tensor): Cosine values of shape (batch, seq, head_dim)
            sin (torch.Tensor): Sine values of shape (batch, seq, head_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key vectors
        """
        head_dim = query.shape[-1]
        # (batch, seq, head_dim) => (batch, 1, seq, head_dim)
        self.cos = cos.unsqueeze(1)
        self.sin = sin.unsqueeze(1)

        # save for backward
        self.query = query
        self.key = key

        # get rotated q and k, (batch, num_head, seq, head_dim)
        self.query_rotated = torch.cat((-query[...,head_dim//2:], query[...,:head_dim//2]), dim=-1)
        self.key_rotated = torch.cat((-key[...,head_dim//2:], key[...,:head_dim//2]), dim=-1)

        # for last dim i<head_dim//2, q_embed[i] = cos * q[i] - sin * q[i+head_dim//2]
        # for last dim i>=head_dim//2, q_embed[i] = cos * q[i] + sin * q[i-head_dim//2]
        # (batch, num_head, seq, head_dim)
        q_embed = self.cos * self.query + self.sin * self.query_rotated
        k_embed = self.cos * self.key + self.sin * self.key_rotated

        return q_embed, k_embed

    def backward(self, grad_q_embed, grad_k_embed):
        """
        Computes the backward pass for RoPE application.

        Parameters:
            grad_q_embed (torch.Tensor): Gradient of rotated query vectors
            grad_k_embed (torch.Tensor): Gradient of rotated key vectors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Gradients for original query, key, cosine, and sine values
        """
        head_dim = grad_q_embed.shape[-1]
        # (batch, num_head, seq, head_dim) => (batch, seq, head_dim)
        grad_cos = (grad_q_embed * self.query + grad_k_embed * self.key).sum(dim=1)
        grad_sin = (grad_q_embed * self.query_rotated + grad_k_embed * self.key_rotated).sum(dim=1)

        # (batch, num_head, seq, head_dim) => (batch, num_head, seq, head_dim)
        grad_query = self.cos * grad_q_embed + self.sin * torch.cat((grad_q_embed[...,head_dim//2:], -grad_q_embed[...,:head_dim//2]), dim=-1)
        grad_key = self.cos * grad_k_embed + self.sin * torch.cat((grad_k_embed[...,head_dim//2:], -grad_k_embed[...,:head_dim//2]), dim=-1)

        return grad_query, grad_key, grad_cos, grad_sin


class CELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        """
        Compute the cross-entropy loss.

        Parameters:
            logits (torch.Tensor): Logits of shape (batch_size, seq, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, seq).

        Returns:
            loss (torch.Tensor): Scalar loss value. (averaged over tokens)
        """
        self.batch_size, self.seq_len, self.vocab_size = logits.shape
        num_tokens = self.batch_size * self.seq_len
        
        # Reshape for easier computation
        self.flattened_logits = logits.reshape(-1, self.vocab_size)  # (tokens, vocab_size)
        self.flattened_labels = labels.reshape(-1)  # (tokens,)
        
        # compute softmax, (tokens, vocab_size)
        max_logits = torch.max(self.flattened_logits, dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(self.flattened_logits - max_logits)
        self.softmax_logits = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
        
        # Compute loss of each token considering the label, (tokens,)
        loss = -torch.log(self.softmax_logits[torch.arange(num_tokens), self.flattened_labels])
        return loss.mean()
    
    def backward(self, grad_loss):
        """
        Compute gradients for cross-entropy loss.
        
        Parameters:
            grad_loss (torch.Tensor): Gradient of the loss scalar
            
        Returns:
            grad_logits (torch.Tensor): Gradient of logits tensor of shape (batch_size, seq, vocab_size)
        """
        num_tokens = self.batch_size * self.seq_len
        # (tokens, )
        grad_loss = (grad_loss / num_tokens) * torch.ones(num_tokens, device=grad_loss.device)
        # (tokens, ) => (tokens, vocab_size)
        one_hot = torch.nn.functional.one_hot(self.flattened_labels, num_classes=self.vocab_size).to(device=grad_loss.device)
        grad_logits = grad_loss.unsqueeze(-1) * (self.softmax_logits - one_hot)
        # (tokens, vocab_size) => (bsz, seqlen, vocab_size)
        grad_logits = grad_logits.reshape(self.batch_size, self.seq_len, -1)

        return grad_logits, self.flattened_labels


class Dropout():
    def __init__(self, p=0.0, training=True):
        self.dropout_prob = p
        self.training = training

    def __call__(self, x):
        """
        Implements dropout regularization.

        Parameters:
            x (torch.Tensor): feature vectors of shape (..., dim)

        Returns:
            torch.Tensor: Feature vectors where elements are randomly dropped (set to 0) with a probability of self.dropout_prob.
        """
        if self.training and self.dropout_prob > 0.:
            dropout_mask = (torch.rand_like(x) > self.dropout_prob)
            x = x * dropout_mask.float()
            # scale the output compensating dropped elements
            return x / (1 - self.dropout_prob)
        return x
