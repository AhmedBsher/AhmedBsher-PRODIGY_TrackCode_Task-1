from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer from directory path
model_path = "./fine_tuned_gpt2"  # Ensure this path is correct and exists
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Example prompt for text generation
prompt = "What is your return policy?"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text
generated = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the generated text
decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Generated Text:", decoded_text)
