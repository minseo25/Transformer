## Implementing Transformer from scratch

2025 Spring - Deep Learning

```bash
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # for gpu
```

## Details

`transformer\abstract.py`
- abstract class Module
- abstract class Optimizer

`transformer\layers.py`
- Linear
- Softmax
- Embedding
- PositionalEmbedding
- RMSNorm (LayerNorm)
- Activation
- RotaryEmbedding
- CELoss

`transformer\blocks.py`
- Attention
- MLP
- FeedForward
- TransformerBlock

`transformer\model.py`
- Transformer

`transformer\optimizers.py`
- SGD
- Momentum
- RMSProp
- Adam

## Useful Sites

softmax: https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

cross entropy loss: https://velog.io/@hjk1996/Cross-Entropy%EC%99%80-Softmax%EC%9D%98-%EB%AF%B8%EB%B6%84

RMSNorm: https://math.stackexchange.com/questions/2882762/is-there-an-easy-way-to-compute-the-jacobian-of-a-normalized-vector
