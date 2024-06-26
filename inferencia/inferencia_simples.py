import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    print("Cuda disponível, rodando na gpu...")
    device = "cuda"
else:
    print("Cuda não disponível, rodando na cpu...")
    device = "cpu"

def gerar_texto(prompt):
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

    prompt = "Detective Emily Carter stood before the imposing gates, her breath forming mist in the chilly air. She adjusted her trench coat, the fabric rustling softly as she surveyed the scene. She had received an invitation to the manor earlier that day—a cryptic message scrawled in elegant handwriting, inviting her to solve a mystery that had plagued the mansion for decades."

    texto_gerado = gerar_texto(prompt)
    print("\n\n----GERADOR DE TEXTO----\n\n")
    print(texto_gerado)

