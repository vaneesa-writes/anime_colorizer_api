import base64
from fastapi import FastAPI, File
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import torch
from base64 import b64encode
from json import dumps, loads
from torchvision import transforms
from generator_model import Generator
from torchvision.utils import save_image


async def load_model():
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generator(in_channels=3, features=64).to(DEVICE)

    checkpoint = torch.load('gen.pth.tar', map_location=DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()
    
    return gen

def transform_image(pil_image):
    
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256, 256)),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    image = data_transform(pil_image).unsqueeze(0)
    
    return image

# gen =  load_model()

def colorize(image):
    
    with torch.no_grad():
        y_fake = gen(image)
        y_fake = y_fake * 0.5 + 0.5  
        save_image(y_fake, "colored_test1.png")
    



app = FastAPI(
    title="ColorMyAnimeAPI",
    description="yoo",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/colorize")
async def detect_return_base64_img(file: bytes = File(...)):
    print("before", type(file))
    pil_image = Image.open(io.BytesIO(file)).convert("RGB")
    print(type(pil_image))
    print(pil_image)
    
    image = transform_image(pil_image)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generator(in_channels=3, features=64).to(DEVICE)

    checkpoint = torch.load('gen.pth.tar', map_location=DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()
    
    with torch.no_grad():
        y_fake = gen(image)
        y_fake = y_fake * 0.5 + 0.5  
        save_image(y_fake, "colored_test1.png")
    
    with open('colored_test1.png', 'rb') as open_file:
        byte_content = open_file.read()
        base64_bytes = b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
        raw_data = {"image": base64_string}
        json_data = dumps(raw_data, indent=2)
        # print(json_data)
    return Response(content=json_data, media_type="image/jpeg")
    
    
    