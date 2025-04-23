FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV TZ=Africa/Lagos
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && \
    apt-get install -y \
        git wget curl ffmpeg tzdata \
        libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m venv $VIRTUAL_ENV

COPY requirements.txt .
RUN $VIRTUAL_ENV/bin/pip install --upgrade pip && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "voice_api:app", "--host", "0.0.0.0", "--port", "8000"]
