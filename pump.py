import torch
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Constants
MODEL_DIR = "/workspace/llava-v1.6-mistral-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model components
print("🔄 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Model loaded on", DEVICE)

# Prompt rules
PROMPT_TEMPLATE = """
You are a degen meme coin creator. Your role is to generate wild, catchy, short meme-style token names and tickers based on a provided tweet text and image.

- Analyze both the tweet and image content.
- Output exactly: {{ "tokenName": "...", "ticker": "..." }}
- No extra text or explanation.
- Ticker must be <=10 chars, ALL CAPS, no spaces.
- If the token name is <=10 chars, use it as the ticker.
- If image has visible text, use that as token name and take first letter of each word as ticker.
"""

def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

def generate_token(tweet, image_url):
    image = load_image_from_url(image_url)

    inputs = processor(
        text=PROMPT_TEMPLATE + f"\nTweet: {tweet}",
        images=image,
        return_tensors="pt"
    ).to(DEVICE, torch.float16)

    output = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract JSON from model output
    try:
        json_str = decoded[decoded.index("{"):decoded.index("}")+1]
        parsed = json.loads(json_str)
        return parsed
    except Exception:
        return {
            "tokenName": "UNKNOWN",
            "ticker": "FAILED"
        }

# --- CLI ---
if __name__ == "__main__":
    tweet = input("Paste the tweet text:\n> ").strip()
    image_url = input("Paste the image URL:\n> ").strip()

    print("\n🚀 Generating Token...")
    result = generate_token(tweet, image_url)
    print("\n✅ Result:\n", json.dumps(result, indent=2))
