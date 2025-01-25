import torch
import torch.nn as nn

from . import implementation


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)


@implementation('orig-llama')
def get_impl(hidden_size, eps, training):
    return LlamaRMSNorm(hidden_size, eps=eps)


@implementation('orig-llama-compiled')
def get_compiled(hidden_size, eps, training):
    return torch.compile(LlamaRMSNorm(hidden_size, eps=eps), mode='max-autotune-no-cudagraphs')
