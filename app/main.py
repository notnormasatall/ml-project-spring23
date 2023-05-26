import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import data_utils
import model_utils
import torch

MODEL_PATH = "improved_model.ckpt"
model = model_utils.SimpleModel.load_from_checkpoint(
    MODEL_PATH, map_location=torch.device('cpu'))

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_wav")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_location = f"{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    _, mask, input, _ = data_utils.get_data(file_location)
    # Pass the data to your model and obtain the MIDI output
    tokens = model.predict(input, mask)
    # Return the MIDI file path for the template
    midi_file = data_utils.generate_midi_file(
        tokens, path=file_location)
    return templates.TemplateResponse("index.html", {"request": request, "midi_file": midi_file})


@app.get("/{filename}")
def download_midi(filename: str):
    return FileResponse(filename, media_type='audio/midi')


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
