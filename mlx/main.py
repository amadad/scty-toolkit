from mlx_lm import load, generate

# Replace 'path_to_converted_model' with the actual path to your converted model
model_path = 'path_to_converted_model'
model, tokenizer = load(model_path)

# Prepare your prompt
prompt = "Your text prompt here"

# Generate SVG output
output = generate(model, tokenizer, prompt=prompt)
print(output)