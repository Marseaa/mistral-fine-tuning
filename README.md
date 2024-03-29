Adaptando o modelo mistral fine tuning usando unsloth (https://github.com/unslothai/unsloth) para ambiente docker.
Para GPU com major_version >= 8

--------------------------------------------------------------------
mistral.py + Dockerfile -> treina dados e cria modelo no huggingface

dentro do diretório "inferencia":
    app.py + dockerfile -> inferencia a partir de modelo criado
--------------------------------------------------------------------