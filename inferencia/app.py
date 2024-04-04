import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# device = "cuda"

def gerar_texto(prompt):
    # carregar modelo salvo no huggingface para realizar a inferencia

    # carrega modelo
    model = AutoModelForCausalLM.from_pretrained("Marseaa/Mistral-7B-Summarization", torch_dtype="auto", trust_remote_code=True)
    
    # carrega tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Marseaa/Mistral-7B-Summarization", trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    outputs = model.generate(**inputs, max_length=24576)
    texto_gerado = tokenizer.batch_decode(outputs)[0]

    return texto_gerado

if __name__ == "__main__":

    while True:
        print("\n\n----GERADOR DE TEXTO PHI-1.5----")
        print("1 - Gerar texto\n2 - Sair")
        op = input("Selecione a opcao: ")

        if op == "1":

            prompt = input("Insira o texto inicial: ")
            texto_gerado = gerar_texto(prompt)
            print("Texto Gerado:")
            print(texto_gerado)

        elif op == "2":
            break

        else:
            print("Opção inválida. Por favor, selecione novamente.")



"""
Modelo com erro:

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# carregar modelo salvo no huggingface para realizar a inferencia

# carrega tokenizer 
tokenizer = AutoTokenizer.from_pretrained("Marseaa/Mistral-7B-Summarization")

# carrega modelo
model = AutoModelForCausalLM.from_pretrained("Marseaa/Mistral-7B-Summarization").to(device)

inputs = tokenizer(
    """

""",
    return_tensors = "pt",
    truncation = True,
    max_length=16384
).to(device)

outputs = model.generate(
    **inputs,
    max_length = 24576,
    num_return_sequences = 1,
    use_cache = True
)

generated_text = tokenizer.batch_decode(outputs, skip_special_tokens = True)

print("\n\nTEXTO GERADO: \n")
print(generated_text)

"""