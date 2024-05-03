from fastapi import FastAPI, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
from pydub.effects import normalize
from datasets import load_dataset
from typing import List, Dict
import errno
import torch
import os
from pydantic import BaseModel
import json
import cv2
from tqdm import tqdm
from pydub import AudioSegment
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutomaticSpeechRecognitionPipeline,
)


app = FastAPI()

# Implementamos modelo de whisper
device = (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # seleccionamos e dispositivo cuda para que sea mas rapiido desde la gpu, si no hay la ponemos desde la cpu
torch_dtype = (
    torch.float16 if torch.cuda.is_available() else torch.float32
)  # reduce la memoria para que sea mas rapido con la gpu

model_id = "openai/whisper-tiny"  # tipo de modelo que usaremos

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,  # estamos cargando el modelo pre-entrenado y dando algunas especificaciones para que trabaje mas optimo
)

model.to(device)  # juntamos para que trabaje con todas las especificaciones

processor = AutoProcessor.from_pretrained(
    model_id
)  # se encarga de tokenizar los datos de entrada y preparar para el modelo

pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]


class File(BaseModel):
    file: UploadFile  # Use UploadFile directly


def recortar_clip(video_path, output_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    clip = VideoFileClip(video_path)

    frame_count = 0
    face_start_frame = None

    try:
        for frame in tqdm(clip.iter_frames(), total=int(clip.fps * clip.duration)):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                if face_start_frame is None:
                    face_start_frame = frame_count
                elif frame_count - face_start_frame >= 5 * clip.fps:
                    clip.subclip(
                        face_start_frame / clip.fps + 10, frame_count / clip.fps + 10
                    ).write_videofile(output_path)
                    break
            else:
                face_start_frame = None

            frame_count += 1
    finally:
        clip.close()


def separate_voice(clip_file_path, voice_output_path):
    # Load the video clip
    clip = VideoFileClip(clip_file_path)

    # Extract the audio from the clip
    audio_clip = clip.audio

    # Save the audio in a separate file
    audio_clip.write_audiofile(voice_output_path, codec="pcm_s16le")

    print("Separated audio saved as", voice_output_path)

    # Close the video clip
    clip.close()


def clean_audio(audio_file):
    # Cargar el archivo de audio
    audio = AudioSegment.from_file(audio_file)

    # Normalizar el volumen del audio para reducir los sonidos de fondo
    cleaned_audio = normalize(audio)

    # Sobrescribir el archivo original con el audio limpio
    cleaned_audio.export(audio_file, format="wav")

    print("Audio limpio guardado como", audio_file)


@app.post("/separate_audio")
async def separate_audio(file: UploadFile):
    try:
        # Guardar el archivo de video
        video_file_path = f"./uploads/{file.filename}"
        with open(video_file_path, "wb") as video_file:
            video_file.write(await file.read())

        # Definir la ruta de salida del audio
        audio_output_path = f"./uploads/audio_{os.path.splitext(file.filename)[0]}.wav"

        # Separar el audio del video
        separate_voice(video_file_path, audio_output_path)

        # Procesar el audio utilizando wav2vec
        result = pipe(audio_output_path)

        # Obtener la ruta del directorio de carga
        upload_dir = os.path.dirname(video_file_path)

        # Definir la ruta de salida del archivo JSON
        output_file_name = (
            os.path.splitext(file.filename)[0].replace(" ", "_") + "_result.json"
        )
        output_file_path = os.path.join(upload_dir, output_file_name)

        # Guardar el resultado en un archivo JSON
        with open(output_file_path, "w") as output_file:
            json.dump(result, output_file)

        # Verificar la generación del archivo
        print("Archivo JSON generado correctamente:", output_file_path)

        return {
            "message": "Procesamiento con wav2vec completado",
            "result_path": output_file_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_clip")
async def extract_clip(file: UploadFile):
    try:
        # Guardar el archivo de video
        video_file_path = f"./uploads/{file.filename}"
        with open(video_file_path, "wb") as video_file:
            video_file.write(await file.read())

        # Obtener el nombre del archivo sin la extensión
        filename_without_extension = os.path.splitext(file.filename)[0]

        # Definir la ruta de salida del clip
        output_path = f"./uploads/clip_{filename_without_extension}.mp4"

        # Recortar el clip del video a 5 segundos
        recortar_clip(video_file_path, output_path)

        # Definir la ruta de salida del audio
        audio_output_path = f"./uploads/_Clip_Audio_{filename_without_extension}.wav"

        # Llamar a la función para separar el audio del clip generado
        separate_voice(output_path, audio_output_path)

        # Llamar a la función para limpiar el audio del clip
        clean_audio(audio_output_path)

        return JSONResponse(
            content={"output_path": output_path, "audio_path": audio_output_path},
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
