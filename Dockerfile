# compatível com unsloth
FROM python:3.9 

# copia arquivos jsonl
COPY instructions.jsonl /app/instructions.jsonl
COPY stories.jsonl /app/stories.jsonl
COPY summaries.jsonl /app/summaries.jsonl

# copia o arquivo mistral.py 
COPY mistral.py /app/mistral.py

# copia o diretório unsloth
COPY unsloth /app/unsloth

# define o diretório de trabalho como /app
WORKDIR /app

# instalação do PyTorch
RUN pip install torch torchvision torchaudio

### unsloth
RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121

RUN pip install "./unsloth[cu121-torch220]"
# RUN pip install "./unsloth[cu121-ampere-torch220]" - aproveita mais a gpu
### unsloth

#outras dependencias
RUN pip install datasets trl transformers 

RUN pip install --no-deps packaging ninja einops xformers trl peft accelerate bitsandbytes

# roda o arquivo
CMD ["python", "mistral.py"]

# docker run --gpus all --name fine_tuning_container -it fine_tuning_image

#---------------------------------------
# instalação flash-attn ainda com erro (dependencia)

# copia o diretorio flash-attn
#COPY flash-attention /app/flash-attention

# instala o flash-attn
#RUN pip install ./flash-attention

#----------------------------------------