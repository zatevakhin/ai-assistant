FROM python:3.11-slim

ARG DEV_USERNAME=ai
ARG DEV_USERID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics,display

RUN apt update \
    && apt install --no-install-recommends -y sudo git neovim curl build-essential python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -rm -d "/home/${DEV_USERNAME}" -s /bin/bash -g root -G sudo -u "${DEV_USERID}" "${DEV_USERNAME}"
RUN echo -n "${DEV_USERNAME}:${DEV_USERNAME}" | chpasswd
RUN echo "${DEV_USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt update \
    && apt install --no-install-recommends -y libopus0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt \
    && rm -rf /tmp/*

COPY --chown=${DEV_USERID}:${DEV_USERID} assistant/ /app/assistant/
RUN chown ${DEV_USERID}:${DEV_USERID} -R /app/

WORKDIR /app/

RUN echo '#!/bin/bash' > /tmp/entrypoint.sh \
    && echo 'set -e' >> /tmp/entrypoint.sh \
    && echo 'export PYTHONPATH="."' >> /tmp/entrypoint.sh \
    && echo 'exec python /app/assistant/main.py $*' >> /tmp/entrypoint.sh \
    && chmod +x /tmp/entrypoint.sh

USER "${DEV_USERNAME}"

ENTRYPOINT ["/tmp/entrypoint.sh"]
