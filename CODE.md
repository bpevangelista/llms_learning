# Code Learning – From Scratch to Efficient Llama2, Mistral and Phi-2 Implementations

### Background

<details>
<summary>RNN (1986) and LSTM (1997) for Sequence Learning</summary>

Sequence learning is used to learn from sequence data such as texts, audio and
video. Sequence data is hard due to <b>order dependency</b> and <b>variable
length</b>. Recurrent Neural Networks (RNN) were state-of-the-art (SOTA) in
sequence learning.

RNNs vs Feed-Forward Networks (FNNs):
[[Ref1](https://www.geeksforgeeks.org/difference-between-feed-forward-neural-networks-and-recurrent-neural-networks/)],
[[Ref2](https://stats.stackexchange.com/questions/2213/whats-the-difference-between-feed-forward-and-recurrent-neural-networks)].

|              | Feed Forward                                | Recurrent                          |
|--------------|---------------------------------------------|------------------------------------|
| Input Length | Fixed                                       | Variable                           |
| Data Flow    | One-way, Top-Down<br>(IN -> hidden0 -> OUT) | Directed Graph<br>(feedback loops) |
| Generation   | Parallel<br>Many Per Iteration              | Sequential<br> One Per Iteration   |

RNNs are GOOD for short-term memory but BAD for long-term (due to precision:
vanishing / exploding gradient). Solution, Long Short-Term Memory (LSTM),
a RNN variant.

LSTM uses recurrent gates to retrieve data over long sequences, thousands of
steps earlier, making them effective for capturing long-term
dependencies
[[Ref](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)].
GRU (2014) is another RNN
variant [[Ref](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)].

However, LSTM and GRU are still limited by dependency, which require large
networks, dependency window size and long processing time. (TODO Add Ref)
</details>

<details>
<summary>Self-Attention in RNN</summary>

Self-Attention has been previously with RNNs
[Ref](https://www.arxiv-vanity.com/papers/1601.06733)

</details>

### Transformer Model (2017) – Multi-Head Self-Attention & MLP/FFN

[Attention Is All You Need Paper](https://arxiv.org/pdf/1706.03762.pdf)

<details>
<summary>Code</summary>

```python
import math
import torch
import torch.nn as nn
 
# Each word (token) is represented by a 4096 dimensional array
embedding_dim = 4096

def multi_head_attention_naive(input_embd: torch.Tensor, num_heads: int) -> torch.Tensor:
    q = nn.Linear(embedding_dim, embedding_dim, False)
    k = nn.Linear(embedding_dim, embedding_dim, False)
    v = nn.Linear(embedding_dim, embedding_dim, False)

    q1 = q(input_embd)
    k1 = k(input_embd)
    v1 = v(input_embd)

    seq_length = input_embd.size(0) 
    head_length = embedding_dim // num_heads

    # view data as [num_heads, seq_length, head_length]
    q2 = q1.view(seq_length, num_heads, head_length).transpose(0, 1)
    k2 = k1.view(seq_length, num_heads, head_length).transpose(0, 1)
    v2 = v1.view(seq_length, num_heads, head_length).transpose(0, 1)

    # q * k_transposed / sqrt(dk)
    dk = q2.size(2)
    qk = q2.matmul(k2.transpose(1, 2)) / math.sqrt(dk)

    # out = softmax(q * k_transposed / sqrt(dk)) * v
    mh_attn = nn.functional.softmax(qk, dim=-1)
    mh_attn_out = mh_attn.matmul(v2)

    # view back as [seq_length, num_heads, head_length] then [seq_length, embedding_dim]
    mh_attn_out = mh_attn_out.transpose(0, 1).reshape(seq_length, embedding_dim)
    return mh_attn_out

input_seq_length = 16
input_embd = torch.rand(input_seq_length, embedding_dim)

multi_head_attention_naive(input_embd, num_heads=16)
```
</details>

TODO Good
Images [Ref](https://bitshots.github.io/Blogs/rnn-vs-lstm-vs-transformer/)
