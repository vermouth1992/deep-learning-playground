import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2").cuda()

input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

tokenizer.apply_chat_template([{
    'role': 'user', 'content': input_text
}])

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))