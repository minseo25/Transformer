import torch

class KVCache():
    """
    A cache that stores the Key and Value states as a list of tensors, one for each layer.
    The expected shape for each tensor is (batch_size, num_heads, seq_len, head_dim).
    """
    def __init__(self, num_hidden_layers):
        self.key_cache = [[] for _ in range(num_hidden_layers)]
        self.value_cache = [[] for _ in range(num_hidden_layers)]
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_id):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_id`.

        Parameters:
            key_states (`torch.Tensor`): The new key states to cache of shape (batch_size, num_heads, seq_len, head_dim).
            value_states (`torch.Tensor`): The new value states to cache of shape (batch_size, num_heads, seq_len, head_dim).
            layer_id (`int`): The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states (batch_size, num_heads, cache_len+seq_len, head_dim)
        """
        # update the number of seen tokens
        if layer_id == 0:
            self._seen_tokens += key_states.shape[2]

        # update the cache
        if self.key_cache[layer_id] == []:
            self.key_cache[layer_id] = key_states
            self.value_cache[layer_id] = value_states
        else:
            # (bsz, nhead, seqlen, hdim) + (bsz, nhead, new_tokens, hdim)
            self.key_cache[layer_id] = torch.cat([self.key_cache[layer_id], key_states], dim=2)
            self.value_cache[layer_id] = torch.cat([self.value_cache[layer_id], value_states], dim=2)

        return self.key_cache[layer_id], self.value_cache[layer_id]
