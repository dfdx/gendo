FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update
RUN apt-get -y install python3 python3-pip git

RUN pip install wheel
RUN pip install torch==2.0.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install accelerate
RUN pip install transformers
RUN pip install datasets
# RUN pip install diffusers[training]
RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install --upgrade "jax[cuda]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax optax
RUN pip install ftfy tensorboard Jinja2
RUN pip install einops

RUN pip install ipython

RUN mkdir -p /root/.cache

RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}' >> /root/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> /root/.bashrc