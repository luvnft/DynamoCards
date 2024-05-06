from fastapi import FastAPI
from pydantic import HttpUrl, BaseModel
from fastapi.middleware.cors import CORSMiddleware
from services.genai import (YoutubeProcessor , GeminiProcessor)
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import sqlite3

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
    
    video_id = uuid.uuid4()
    url_id = f"{str(request.youtube_link)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    video_url = str(request.youtube_link)
    if not is_video_url_cached(video_url):
        print("---------------------ENTERED NOT IN DATABASE ----------------")
        processor = YoutubeProcessor(genai_processor=genai_processor)

        result = processor.retrieve_youtube_documents(str(request.youtube_link),verbose = True)

        # summary = genai_processor.generate_document_summary(result,verbose=True)

        #Find Key Concepts : 
        key_concepts = processor.find_key_concepts(result,verbose=True)
        
        key_concept_list = {}
        processed_concepts = []
        for concept_dict in key_concepts:
            for key,value in concept_dict.items():
                key_concept_list[key] = value
                processed_concepts.append((key,value))


        key_concepts_result = [{key:value} for key,value in concept_dict.items()]
        save_to_database(video_id,url_id,processed_concepts,str(request.youtube_link))
        return {
            "key_concepts" : key_concepts_result
        }
    else:
        print("---------------ENTERED IN DATABASE -------------")
        key_concepts = retrieve_key_concepts_from_db(video_url)
        key_concept_list = [{concept[0]:concept[1]} for concept in key_concepts]
        return {
            "key_concepts": key_concept_list
            }
    
def is_video_url_cached(video_url):
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'video_analysis' ") 
    table_exists = cursor.fetchone() is not None
    if not table_exists:
        conn.close()
        return False
    cursor.execute("SELECT COUNT(*) FROM video_analysis WHERE videourl = ?",(video_url,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def retrieve_key_concepts_from_db(video_url):
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute("SELECT concept,definition FROM video_analysis where videourl = ? ",(video_url,))
    key_concepts = cursor.fetchall()
    print("-------------------------retrieving key concepts ---------------",key_concepts)
    for concept in key_concepts:
        print("--------------CONCEPT : --------------",concept[0])
        
        print("--------------DEFINITION : --------------",concept[1])
    conn.close()
    return key_concepts

@app.get("/root")
def health():
    return {"status":"ok"}

def save_to_database(video_id, url_id, processed_concepts,video_url):
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()

    cursor.execute(''' CREATE TABLE IF NOT EXISTS video_analysis (
                   video_id TEXT,
                   url_id TEXT,
                   videourl TEXT,
                   concept TEXT,
                   definition TEXT  )
                    ''')
    for concept,definition in processed_concepts:
        print("------------CONCEPT ---------",concept)
        print("------------------defintion-----------",definition)
        cursor.execute("INSERT INTO video_analysis VALUES (?, ?, ?, ?, ?)",(str(video_id),url_id,video_url,concept,definition))

    conn.commit()
    conn.close()

