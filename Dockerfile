FROM python:3.8

# copia o arquivo mistral.py 
COPY mistral.py /app/mistral.py

# define o diretório de trabalho como /app
WORKDIR /app

# instalação do PyTorch
RUN pip install torch torchvision torchaudio

### unsloth
RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121

RUN pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install "unsloth[cu118-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
### unsloth

RUN pip install datasets trl transformers 

RUN pip install --no-deps packaging ninja einops xformers trl peft accelerate bitsandbytes

#sem flash-attn

CMD ["python", "mistral.py"]
