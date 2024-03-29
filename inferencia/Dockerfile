# Dockerfile para fazer inferencia com modelo salvo no huggingface

FROM python:3.9 

COPY app.py /app/app.py

WORKDIR /app

# instalacao das dependencias
RUN pip install transformers

# instalacao pytotch
RUN pip install torch torchvision torchaudio

# executando a inferencia
CMD ["python", "app.py"]

 