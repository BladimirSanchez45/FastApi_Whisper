from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
import cv2
from tqdm import tqdm
from pydub import AudioSegment
from pydub.effects import normalize
import os
from pydantic import BaseModel

app = FastAPI()


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
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Normalize the audio volume to reduce background noise
    cleaned_audio = normalize(audio)

    # Overwrite the original file with the cleaned audio
    cleaned_audio.export(audio_file, format="wav")

    print("Clean audio saved as", audio_file)


@app.post("/separate_audio")
async def separate_audio(file: UploadFile):
    try:
        # Save the video file
        video_file_path = f"./uploads/{file.filename}"
        with open(video_file_path, "wb") as video_file:
            video_file.write(await file.read())

        # Define the output audio path
        audio_output_path = f"./uploads/audio_{os.path.splitext(file.filename)[0]}.wav"

        # Separate audio from the video
        separate_voice(video_file_path, audio_output_path)

        # Clean the audio
        clean_audio(audio_output_path)

        return {
            "message": "Audio separated and cleaned successfully",
            "audio_path": audio_output_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_clip")
async def extract_clip(file: UploadFile):
    try:
        # Save the video file
        video_file_path = f"./uploads/{file.filename}"
        with open(video_file_path, "wb") as video_file:
            video_file.write(await file.read())

        # Get the filename without extension
        filename_without_extension = os.path.splitext(file.filename)[0]

        # Define the output clip path
        output_path = f"./uploads/clip_{filename_without_extension}.mp4"

        # Trim the video clip to 5 seconds
        recortar_clip(video_file_path, output_path)

        # Define the output audio path
        audio_output_path = f"./uploads/_Clip_Audio_{filename_without_extension}.wav"

        # Call the function to separate audio from the generated clip
        separate_voice(output_path, audio_output_path)

        # Call the function to clean the audio from the clip
        clean_audio(audio_output_path)

        return JSONResponse(
            content={"output_path": output_path, "audio_path": audio_output_path},
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
