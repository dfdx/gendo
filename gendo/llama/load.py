import os
import json
import jax
import jax.numpy as jnp
import torch
from tqdm import tqdm
from gendo.llama.convert import pt2jax
from gendo.llama.model import Transformer, ModelArgs
# from gendo.llama.model_pt import Transformer as PtTransformer


def convert_to_nested(flat: dict):
    """
    Convert flat structure of PyTorch state dict to
    nested structure of JAX params.
    """
    def join_numbers(key_seq):
        # ["layer", "0", ...] -> ["layer_0", ...]
        assert len(key_seq) > 0 and not key_seq[0].isnumeric()
        out = []
        for key in key_seq:
            if key.isnumeric():
                out[-1] = out[-1] + "_" + key
            else:
                out.append(key)
        return out
    nested = {}
    for key, val in flat.items():
        key_seq = key.split(".")
        key_seq = join_numbers(key_seq)
        dct = nested
        for subkey in key_seq[:-1]:
            if subkey not in dct:
                dct[subkey] = {}
            dct = dct[subkey]
        dct[key_seq[-1]] = val
    return nested


def convert_linear(state: dict):
    assert "bias" not in state, "Convertion of bias in torch.nn.Linear is not supported yet"
    return {
        "kernel": pt2jax(state["weight"].T)
    }


def convert_attention(state: dict):
    assert len(state.keys()) == 4
    return {
        "wq": convert_linear(state["wq"]),
        "wk": convert_linear(state["wk"]),
        "wv": convert_linear(state["wv"]),
        "wo": convert_linear(state["wo"]),
    }


def convert_feed_forward(state: dict):
    assert len(state.keys()) == 3
    return {
        "w1": convert_linear(state["w1"]),
        "w2": convert_linear(state["w2"]),
        "w3": convert_linear(state["w3"]),
    }

def convert_attention_norm(state: dict):
    return {
        "weight": pt2jax(state["weight"]),
    }


def convert_ffn_norm(state: dict):
    return {
        "weight": pt2jax(state["weight"]),
    }


def convert_layer(state: dict):
    return {
        "attention": convert_attention(state["attention"]),
        "feed_forward": convert_feed_forward(state["feed_forward"]),
        "attention_norm": convert_attention_norm(state["attention_norm"]),
        "ffn_norm": convert_ffn_norm(state["ffn_norm"]),
    }


def convert_tok_embeddings(state: dict):
    return {
        "embedding": pt2jax(state["weight"]),
    }


def convert_norm(state: dict):
    return {
        "weight": pt2jax(state["weight"]),
    }


def convert_transformer(state: dict):
    layers = {k: convert_layer(state[k]) for k, v in state.items() if k.startswith("layers_")}
    return {
        "tok_embeddings": convert_tok_embeddings(state["tok_embeddings"]),
        "norm": convert_norm(state["norm"]),
        "output": convert_linear(state["output"]),
        **layers,
    }


def load_from_torch(model_dir: str, vocab_size: int, max_batch_size=None, max_seq_len=None):
    model_path = os.path.join(model_dir, "consolidated.00.pth")
    args_path = os.path.join(model_dir, "params.json")
    with open(args_path) as fp:
        args = ModelArgs(**json.load(fp))
        args.vocab_size = vocab_size
        if max_batch_size:
            args.max_batch_size = max_batch_size
        if max_seq_len:
            args.max_seq_len = max_seq_len
    model = Transformer(args)
    # init variables including caches
    variables = model.init(jax.random.PRNGKey(0), jnp.asarray([[22172, 3186]]), 0)
    cache = variables["cache"]
    state_dict = torch.load(model_path)
    nested_state_dict = convert_to_nested(state_dict)
    variables = {
        "params": convert_transformer(nested_state_dict),
        "cache": cache,
    }
    return model, variables




def main():
    from gendo.llama.tokenizer import Tokenizer
    import jax
    import jax.numpy as jnp

    tokenizer_path = "/data/llama/tokenizer.model"
    tokenizer = Tokenizer(model_path=tokenizer_path)
    vocab_size = tokenizer.n_words

    model_dir = "/data/llama/llama-2-7b"
    model, variables = load_from_torch(model_dir, vocab_size, max_batch_size=1, max_seq_len=512)

    rng = jax.random.PRNGKey(81)
    tokens = tokenizer.encode("frankenstein walks into a bar", False, False)
    tokens = jnp.asarray(tokens).reshape(1, -1)
    model.apply(variables, tokens, 0, mutable=["cache"])