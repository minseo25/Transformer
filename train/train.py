import torch
from transformer.model import Transformer
from transformer.optimizers import SGD, Momentum, RMSProp, Adam
import numpy as np
from utils.tokenizer import RegexBPETokenizer

class Shakespeare():
    """
    !! Alert !!
    This class provides an abstracted dataloader for the tiny shakespeare dataset.
    However, for the fast training, the starting token index of the batch is restricted to the multiple of 100.
    For example, for a given dataset with a length 1M,
    the number of total given batch is 1M / 100 = 10k.
    Please keep this in mind if you want to adopt this dataloader for other purposes.
    """
    def __init__(self, device, vocab_size):
        with open("train\\tiny_shakespeare.txt") as f:
            data = f.read()
        print("dataset length:", len(data))

        self.device = device

        self.tokenizer = RegexBPETokenizer()
        self.tokenizer.train(data[:30000], vocab_size)

        self.trainset = self.tokenizer.encode(data[:30000])
        self.trainset = torch.tensor(self.trainset, dtype=torch.long, device=device)
        self.trainset_size = len(self.trainset)

        self.testset = self.tokenizer.encode(data[30000:])
        self.testset = torch.tensor(self.testset, dtype=torch.long, device=device)
        self.testset_size = len(self.testset)

    def dataloader(self, split, batch_size, seq_len):
        if split == 'train':
            dataset = self.trainset
        else:
            dataset = self.testset
        start_indices = np.arange(0, len(dataset) - seq_len, seq_len)
        np.random.shuffle(start_indices)

        for i in range(0, len(start_indices), batch_size):
            input_idx = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)
            labels = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)

            for j in range(i, min(i + batch_size, len(start_indices))):
                start = start_indices[j]
                input_idx[j-i] = dataset[start:start+seq_len]
                labels[j-i] = dataset[start+1:start+seq_len+1]

            yield input_idx, labels


vocab_size = 256 + 256
device = 'cuda:0'
shakespeare = Shakespeare(device, vocab_size)

batch_size, seq_len, lr = 64, 100, 0.03

optimizers = {
    'SGD': SGD,
    'Momentum': Momentum,
    'RMSProp': RMSProp,
    'Adam': Adam,
}

train_log = {
    'SGD': [],
    'Momentum': [],
    'RMSProp': [],
    'Adam': [],
}

test_log = {
    'SGD': [],
    'Momentum': [],
    'RMSProp': [],
    'Adam': [],
}

for name, opt_class in optimizers.items():
    model = Transformer(4, vocab_size, 128, 16).to(device)
    optimizer = opt_class(model, lr=lr, wd=0.001)

    print(f"Optimizer: {name}")

    for i in range(50):
        train_loss, test_loss = 0, 0

        for input_idx, label in shakespeare.dataloader('train', batch_size, seq_len):
            logits, loss = model(input_idx, label)
            model.backward(torch.zeros_like(logits), torch.ones_like(loss))
            optimizer.step()

            pred = torch.argmax(logits, dim=-1)
            train_loss += loss.item() * label.numel()

        for input_idx, label in shakespeare.dataloader('test', batch_size, seq_len):
            logits, loss = model(input_idx, label)

            pred = torch.argmax(logits, dim=-1)
            test_loss += loss.item() * label.numel()

        train_loss = train_loss / shakespeare.trainset_size
        test_loss = test_loss / shakespeare.testset_size

        if i % 10 == 0:
            print(f"[{i:3d}/50], train loss {train_loss:.4f}, test loss {test_loss:.4f}")

        train_log[name].append(train_loss)
        test_log[name].append(test_loss)
    print()
