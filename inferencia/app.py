from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# carregar modelo salvo no huggingface para realizar a inferencia

# carrega tokenizer 
tokenizer = AutoTokenizer.from_pretrained("Marseaa/Mistral-7B-Summarization")

# carrega modelo
model = AutoModelForCausalLM.from_pretrained("Marseaa/Mistral-7B-Summarization").to(device)

inputs = tokenizer(
    """Summarize the following story in my style

### Story
Chapter 1: The Whispering Peaks
Dr. Elena Vasquez peered out of the small airplane window, her eyes tracing the outline of the jagged mountains that loomed below. The Andean sun bathed the landscape in a harsh, unforgiving light, revealing a world untouched and unexplored. She adjusted her glasses, her heart racing with the anticipation of setting foot on a land that had, until now, been nothing more than a myth.

For years, Elena had scoured ancient texts and interviewed locals, piecing together tales of the "Whispering Peaks," a remote mountain range shrouded in mystery and rumored to be the home of undiscovered flora and fauna. As a botanist with a penchant for adventure, she was drawn to the unknown, and the possibility of uncovering new species was a call she couldn't resist.


### Summary

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

