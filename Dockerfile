FROM nvidia/cuda:11.6.2-runtime-ubuntu18.04 as cuda-base

SHELL ["/bin/bash", "-c"]

ENV HOME=/root
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV BASHRC=/root/.bashrc

ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

COPY pyenv.tar.gz /pyenv.tar.gz
RUN mkdir -p $PYENV_ROOT && cd $PYENV_ROOT && tar xvzf /pyenv.tar.gz \
    && echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> $BASHRC \
    && echo 'eval "$(pyenv init --path)"' >> $BASHRC \
    && echo 'eval "$(pyenv virtualenv-init -)"' >> $BASHRC \
    && source $BASHRC


ENV PYTHON_VERSION=3.9.12

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=linux

ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8

RUN apt update && apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gpg \
        gpg-agent \
        less \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        llvm \
        locales \
        tk-dev \
        tzdata \
        unzip \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        zstd \
    && sed -i "s/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g" /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt clean

RUN pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && rm -rf /tmp/*

RUN mkdir -p ~/.cache/huggingface
RUN pip install --no-cache-dir ipython pytest



FROM cuda-base

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

RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}' >> /root/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> /root/.bashrc