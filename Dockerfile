FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04

RUN apt-get update
RUN apt-get -y install python3 python3-pip git

RUN pip install wheel
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install transformers
# RUN pip install diffusers[training]
RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install --upgrade "jax[cuda]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax optax
RUN pip install ftfy tensorboard Jinja2

RUN pip install ipython

RUN mkdir -p /root/.cache