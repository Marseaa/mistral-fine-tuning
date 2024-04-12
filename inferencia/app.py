import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    print("Cuda dispobível, rodando na gpu...")
    device = "cuda"
else:
    print("Cuda não disponível, rodando na cpu...")
    device = "cpu"

def gerar_texto(prompt):
    os.system("clear")
    print("\n\nGerando texto a partir do input: ")
    print(prompt)
    print("\n\n")

    # carregar modelo salvo no huggingface para realizar a inferencia

    # carrega modelo
    model = AutoModelForCausalLM.from_pretrained("Marseaa/Mistral-7B-Summarization", torch_dtype="auto", trust_remote_code=True).to(device)
    
    # carrega tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Marseaa/Mistral-7B-Summarization", trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Mover para a GPU
    outputs = model.generate(**inputs, max_length=300).to(device) # num max de caracteres -> pode aumentar
    texto_gerado = tokenizer.batch_decode(outputs)[0]


    return texto_gerado

if __name__ == "__main__":

    while True:
        os.system("clear")
        print("\n\n----GERADOR DE TEXTO PHI-1.5----")
        print("1 - Gerar texto\n2 - Sair")
        op = input("Selecione a opcao: ")

        if op == "1":
            os.system("clear")
            print("\n\n----GERADOR DE TEXTO PHI-1.5----")
            prompt = input("Insira o texto inicial: ")

            texto_gerado = gerar_texto(prompt)
            print("\n\nGeracao de texto concluida...")
            print("Pressione Enter para visualizar o texto gerado...")
            input()
            os.system("clear")

            print("\n\n----GERADOR DE TEXTO PHI-1.5----")
            print("\n\nTexto inserido: ")
            print(prompt)
            print("\n\nTexto Gerado:\n")
            print(texto_gerado)
            print("Pressione Enter para continuar...")
            input()

        elif op == "2":
            os.system("clear")
            print("\n\nSaindo...\n\n")
            break

        else:
            print("Opção inválida. Por favor, selecione novamente.")


