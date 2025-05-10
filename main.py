from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from io import BytesIO
import uvicorn
import os
import uuid

# Load the model
model = tf.keras.models.load_model("brain_tumor_multiclass_cnn.h5")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Image transformation
def transform_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Invalid image file: " + str(e))
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Serve HTML
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())

# Prediction + Annotated Image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = transform_image(image_bytes)
        predictions = model.predict(input_tensor)[0]
        pred_idx = np.argmax(predictions)
        pred_label = CLASS_NAMES[pred_idx]

        # Re-open image to annotate
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Prediction: {pred_label}", fill="red", font=font)

        y_offset = 30
        for i, cls in enumerate(CLASS_NAMES):
            text = f"{cls}: {predictions[i]*100:.2f}%"
            draw.text((10, y_offset), text, fill="blue", font=font)
            y_offset += 20

        # Save annotated image
        image_id = str(uuid.uuid4())[:8]
        output_filename = f"prediction_{image_id}.png"
        output_path = f"static/{output_filename}"
        image.save(output_path)

        return {
            "prediction": pred_label,
            "probabilities": {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))},
            "annotated_image_url": f"/static/{output_filename}"
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": "An unexpected error occurred: " + str(e)}

# Optional: download route
@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join("static", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename=filename)
    return {"error": "File not found"}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
# To run the app, use the command: uvicorn main:app --reload
# Ensure you have the required packages installed: