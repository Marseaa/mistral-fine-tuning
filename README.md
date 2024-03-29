Adaptando o modelo mistral fine tuning usando unsloth (https://github.com/unslothai/unsloth) para ambiente docker.
Para GPU com major_version >= 8

--------------------------------------------------------------------
mistral.py + Dockerfile -> treina dados e cria modelo no huggingface

dentro do diretório "inferencia": (modelo já criado, pode rodar a inferencia direto)
    app.py + dockerfile -> inferencia a partir de modelo criado
--------------------------------------------------------------------