This is an unfinished experiment on translating Llama 2 to JAX. The code works properly and is covered with a number of tests. However, without JIT compilation it performs quite poorly, and JIT doesn't like the dynamic `start_pos` argument, which is essential to the cache mechanics of the Llama. `example.py` contains an example of a single call with JIT (`start_pos == 0`) and `generation.py` contains current state of the full generation pipeline. Feel free to play around and improve.