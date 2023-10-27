import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def generate_response(
    instruction: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    do_sample: bool = True,
    max_new_tokens: int = 256,
    top_p: float = 0.92,
    top_k: int = 0,
    **kwargs,
) -> str:
    input_ids = tokenizer(
        PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt"
    ).input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=end_key_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]

        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None


def main():
    # from transformers import FlaxAutoModelForCausalLM
    # from transformers.models.gptj.modeling_flax_gptj import FlaxGPTJForCausalLMModule
    # FlaxAutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True)
    # FlaxGPTJForCausalLMModule()

    tokenizer = AutoTokenizer.from_pretrained(
        "databricks/dolly-v1-6b", padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True
    )
    # Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"

    text = """
    Katty has always wanted to be a programmer. Thus, for Christmas she asked her parents to buy her a personal computer so she could write programs in Python.
    But her parents had other plans. Her dad, who was a lowyer, wanted Katty to go to the college and get a degree in biology.
    On Christam eve, Katty realized her dream won't come true if she stays home. And so she leaves.
    Early in the morning, Katty wrote a letter to her parents: "Dear mom and dad, thank you for all the cookies, but today I'm leaving to make my dream come true".

    Then she took her bag and quickly ran to the nearest town.

    But before she even left her own town, she noticed a railway. "Why haven't I seen it before?" - she asked herself. - "But it is convenient - I'll go by train".

    And so Katty jumped into the nearest train car.
    """
    out = generate_response(
        f"summarize in less than 100 words: {text}", model=model, tokenizer=tokenizer
    )
    print(out)
