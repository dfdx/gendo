# This is an example of a single call of a transformer (using randomly initialized parameters).
# See generation.py for a an example of generation using this transformer.
# Note: you will need Llama weights, see the official instruction:
# https://github.com/facebookresearch/llama#download

from functools import partial
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from gendo.llama.model import Transformer, ModelArgs
from gendo.llama.tokenizer import Tokenizer

TOKENIZER_PATH = "/data/llama/tokenizer.model"


def main():
    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
    args.vocab_size = tokenizer.n_words
    tokens = tokenizer.encode("frankenstein walks into a bar", False, False)
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    variables = tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), variables)
    # note: we make start_pos static to make JIT happy (specifically, in jnp.triu),
    # but it leads to re-compilation each new value; I'd be happy to find a better way
    jit_apply = partial(jax.jit, static_argnums=(2,), static_argnames=("mutable",))(model.apply)
    # cache is updated during the call, so we get both - logits and updated cache values
    logits, _variable_updates = jit_apply(variables, tokens, 0, mutable=("cache",))
    print(logits)


if __name__ == "__main__" and "__file__" in globals():
    main()