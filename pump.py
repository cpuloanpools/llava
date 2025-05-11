import torch
from PIL import Image
from transformers import TextStreamer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates
import requests
from io import BytesIO

# ==== Settings ====
model_path = "/workspace/llava-v1.6-mistral-7b-hf"
image_url = input("Paste image URL:\n> ").strip()
tweet = input("Paste tweet text:\n> ").strip()

# ==== Load Image ====
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# ==== Load Model ====
tokenizer, model, processor, context_len = load_pretrained_model(
    model_path, model_path, model_name="llava-v1.6-mistral-7b-hf"
)
model.eval()
model.cuda()

# ==== Prepare Input ====
image_tensor = process_images([image], processor, model.config).to(dtype=torch.float16, device="cuda")
prompt_template = """
You are a meme coin creator. Generate a catchy token name and ticker based on this tweet and image.
Tweet: {tweet}
Output in JSON format like: {{ "tokenName": "NAME", "ticker": "TICKER" }}
""".strip()

conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], prompt_template.format(tweet=tweet))
conv.append_message(conv.roles[1], None)

input_ids = tokenizer_image_token(
    conv.get_prompt(), tokenizer, processor, image_tensor
)

# ==== Generate ====
output_ids = model.generate(
    input_ids=input_ids,
    images=image_tensor,
    do_sample=False,
    temperature=0.2,
    max_new_tokens=120,
    use_cache=True
)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n✅ Output:\n", output)
