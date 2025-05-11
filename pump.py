import requests
import json
from io import BytesIO
from PIL import Image
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


# Set up LLaVA model
MODEL_PATH = "/workspace/llava-v1.6-mistral-7b-hf"  # update if stored elsewhere
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name="llava"
)
model.eval()

# Your meme prompt
MEME_PROMPT = """
You are a degen meme coin creator. Your role is to generate wild, catchy, short meme-style token names and tickers based on a provided tweet text and image URL. You prioritize humor, virality, and fitting into the meme coin culture. You must strictly follow these rules: 

- Input will be Tweet text and Image URL
- Analyze both the tweet text and The Image along with the visible text inside the image.
- The vibe of the tweet and image (funny, hype, crypto culture, absurd) must inspire the token name.
- Output must be exactly two things: Token Name and Ticker.
- No explanation, no extra commentary.
- If the token name has multiple words, include spaces between them.
- If a specific object is named in the tweet or image, set the ticker as that object’s name.
- If a prominent or famous person is shown in the image, that person becomes the main object.
- If the image has unique visible text, use that text as the token name, and create the ticker by taking the first letter of each word.
- Remember If the token name is 10 characters or fewer, use the same name as the ticker Without spaces.
- The ticker must always reflect the main highlight of the tweet or image.
- Ticker cannot have spaces and it cannot be more than 10 letters
- Output should in the form of a JSON object
{ "tokenName": "generated_token_name", "ticker": "generated_ticker_name" }

You must stick exactly to these rules without exception. Prioritize wildness and absurdity, but always stay accurate to the text and image provided.

Now, behave exactly according to the above instructions every time you are given a tweet text and/or an image URL.
"""  # (keep the full prompt as in your original)

# Download image and convert to tensor
def image_url_to_tensor(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = process_images([img], image_processor, model.config)[0].unsqueeze(0).to(model.device)
    return tensor

# Run LLaVA generation
def ask_llava(prompt, image_tensor):
    formatted_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    input_ids = tokenizer_image_token(formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(model.device)
    output_ids = model.generate(**input_ids, images=image_tensor, max_new_tokens=200)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# CLI flow
def main():
    print("🧠 Meme Token Generator (LLaVA-Next Mistral)")
    tweet = input("Paste the tweet text:\n> ").strip()
    image_url = input("Paste the image URL:\n> ").strip()
    print("\n⏳ Generating...\n")

    try:
        image_tensor = image_url_to_tensor(image_url)
        prompt = f"{MEME_PROMPT}\nTweet: {tweet}"
        result = ask_llava(prompt, image_tensor)
        print("✅ Output:\n" + result.strip())
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
