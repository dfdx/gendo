# devel version with DNN is required for JIT compilation in some cases
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build-base


SHELL ["/bin/bash", "-c"]

RUN apt-get update
# RUN apt-get -y install python3 python3-pip git

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


ENV PYTHON_VERSION=3.10.6

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



RUN pip install wheel

# jax needs to be installed first because otherwise torch dependencies will force older version of jax
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax optax

RUN pip install sentencepiece

# fix torch version because updating it often leads to painful results
# RUN pip install torch==2.0.0 torchvision lightning
RUN pip install transformers accelerate datasets
RUN pip install einops


RUN mkdir -p /root/.cache

# ensure libraries see CUDA
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}' >> /root/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> /root/.bashrc


FROM build-base AS build-dev

RUN pip install ipython mypy black

CMD ["echo", "Let's talk about it"]