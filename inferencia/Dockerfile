# Dockerfile para fazer inferencia com modelo salvo no huggingface

FROM python:3.9 

COPY app.py /app/app.py

WORKDIR /app

# instalacao das dependencias
RUN pip install transformers

# instalacao pytotch
RUN pip install torch torchvision torchaudio

RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121

# define a variável de ambiente para especificar que o Docker usará a GPU
ENV NVIDIA_VISIBLE_DEVICES all

# executando a inferencia
CMD ["python", "app.py"]

# docker run -v cache_volume:/pip-cache -it --network redeprincipal --gpus all --name test_inference_container teste_inference_image:tag_teste
# docker start -ai test_inference_containercls
