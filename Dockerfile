FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN pip install wheel
RUN pip install transformers
# RUN pip install diffusers[training]
RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install --upgrade "jax[cuda]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax optax
RUN pip install ftfy tensorboard Jinja2

RUN pip install ipython

RUN mkdir -p /root/.cache

# RUN echo 'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/' >> /root/.bashrc
# RUN echo 'PATH=${PATH}:/usr/local/cuda/bin' >> /root/.bashrc
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/
ENV PATH=${PATH}:/usr/local/cuda/bin