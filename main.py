import io
import uvicorn
from fastapi import File, UploadFile, Request, FastAPI
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import pipeline
import tensorflow as tf
    
app = FastAPI()
templates = Jinja2Templates(directory="assets")

def load_model():
    return pipeline(model="JuanMa360/room-classification")

def preprocess_image(img):
    img = img.resize((100, 100))
    x = tf.keras.utils.img_to_array(img)
    tf.keras.applications.efficientnet.preprocess_input(x)

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("assets/uploaded/" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        
    loadedImage = Image.open(io.BytesIO(contents))

    model = load_model()
    
    preprocess_image(loadedImage)
    prediction = model.predict(loadedImage)
    return {"result": prediction}

if __name__ == "__main__":
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)