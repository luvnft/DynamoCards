from fastapi import FastAPI
from pydantic import HttpUrl, BaseModel
from fastapi.middleware.cors import CORSMiddleware
from services.genai import (YoutubeProcessor , GeminiProcessor)
import os
from dotenv import load_dotenv

load_dotenv()

class VideoAnalysisRequest(BaseModel):
    youtube_link : HttpUrl

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

genai_processor = GeminiProcessor(model_name = "gemini-pro",project=os.getenv("PROJECT_NAME"))

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    
    processor = YoutubeProcessor(genai_processor=genai_processor)

    result = processor.retrieve_youtube_documents(str(request.youtube_link),verbose = True)

    # summary = genai_processor.generate_document_summary(result,verbose=True)

    #Find Key Concepts : 
    key_concepts = processor.find_key_concepts(result,verbose=True)
    
    return {
        "key_concepts" : key_concepts
    }

@app.get("/root")
def health():
    return {"status":"ok"}