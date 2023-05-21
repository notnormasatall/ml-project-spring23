from pathlib import Path
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import os
import tempfile
import subprocess
import data_utils
import model_utils

MODEL_PATH = "linear_norm_t5.ckpt"
model = model_utils.SimpleModel.load_from_checkpoint(MODEL_PATH)

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

    output, mask, input, output_mask = data_utils.get_data(file_location)

    # Pass the data to your model and obtain the MIDI output
    tokens = model_utils.predict_tokens(
        model, None, mask, input, None)

    print(tokens)
    # Return the MIDI file path for the template
    midi_file = "MIDI-Unprocessed_02_R2_2008_01-05_ORIG_MID--AUDIO_02_R2_2008_wav--4.midi"
    # data_utils.generate_midi_file(
    #     tokens, path=f"MIDI-Unprocessed_02_R2_2008_01-05_ORIG_MID--AUDIO_02_R2_2008_wav--4.midi")
    return templates.TemplateResponse("index.html", {"request": request, "midi_file": midi_file})


@app.get("/midi/{filename}")
def download_midi(filename: str):
    print()
    return FileResponse(filename, media_type='audio/midi')


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
