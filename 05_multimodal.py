"""
05_multimodal.py: Vision + text input (multimodal)
RUN:
    1. Place any .jpg file in the project root and name it: image.jpg
    2. uv run 05_multimodal.py

WHAT IT DOES:
    - Reads an image from disk and encodes it as Base64.
    - Sends image + text question together in a single message.
    - The model reads both and answers based on what it sees.

EXPECTED OUTPUT:
    (A description of whatever is in image.jpg)

WHY BASE64?
    HTTP/JSON can't carry raw binary data directly.
    Base64 converts binary to an ASCII string.
    It's then sent as a data URI: "data:image/jpeg;base64,..."

MODEL:
    Qwen/Qwen2.5-VL-72B-Instruct. "VL" stands for Vision-Language.
    Llama 3.3 is text-only; a VL model is required for image input.
"""

import os
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# VL (Vision-Language) model: understands both text and images.
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

image_path = "image.jpg"  # place a .jpg file here before running

# Step 1: read the image as raw bytes.
# Step 2: encode bytes to Base64 ASCII.
# decode("utf-8") converts bytes to a Python string.
with open(image_path, "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

# A single HumanMessage that carries both text and image.
# "type": "image_url" signals to the model that this content is an image.
# The data URI embeds the image directly in the request.
message = HumanMessage(
    content=[
        {"type": "text", "text": "What do you see in this image? Describe it briefly."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ]
)

response = llm.invoke([message])
print(response.content)
