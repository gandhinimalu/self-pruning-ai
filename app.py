from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

from model import Net
from rag import retrieve

app = FastAPI()

model = Net()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out,1).item()

    return {"prediction": classes[pred]}

@app.get("/explain")
def explain(q: str):
    return {"query": q, "context": retrieve(q)}