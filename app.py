import os
import cv2
import base64
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

MONGO_URI = os.getenv("mongodb+srv://technglobalhosting_db_user:RhbVhLnehkcVIBa1@miniproject.pg2mobb.mongodb.net/?appName=miniproject", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "photosort_db")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db["photos"]

app = FastAPI(title="PhotoSort Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassificationUpdate(BaseModel):
    category: str = Field(..., pattern="^(good|review|poor)$")

def analyze_image_cv(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file or unsupported format.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = round(float(laplacian_var), 2)
    mean_lum = round(float(np.mean(gray)), 2)

    issues = []
    score = 90

    if sharpness_score < 60.0:
        issues.append("Out of Focus")
        score -= 40
    elif sharpness_score < 120.0:
        issues.append("Soft Focus")
        score -= 15

    if mean_lum < 45.0:
        issues.append("Underexposed")
        score -= 30
    elif mean_lum > 215.0:
        issues.append("Blown Highlights")
        score -= 30

    score = max(10, min(99, int(score)))

    if score >= 75:
        category = "good"
    elif score >= 50:
        category = "review"
    else:
        category = "poor"

    if not issues:
        issues = ["Sharp Focus", "Balanced Exposure"]

    return {
        "score": score,
        "category": category,
        "sharpness": sharpness_score,
        "exposure": mean_lum,
        "issues": issues,
        "resolution": f"{img.shape[1]}x{img.shape[0]}"
    }

def format_doc(doc):
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "index.html not found in root folder"}

@app.get("/api/health")
async def health_check():
    try:
        await client.admin.command('ping')
        db_status = "connected"
    except Exception as e:
        db_status = f"unreachable: {str(e)}"
    return {"status": "ok", "mongodb": db_status}

@app.post("/api/photos/analyze")
async def analyze_and_save_photo(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    
    # Check 16MB MongoDB Atlas BSON document threshold
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max size is 15MB.")

    try:
        cv_results = analyze_image_cv(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

    # Convert binary to base64 Data URL so it is fully stored and viewable
    encoded_image = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{file.content_type};base64,{encoded_image}"

    record = {
        "filename": file.filename,
        "image_data": data_url,  # <--- Storing the actual image in MongoDB
        "score": cv_results["score"],
        "category": cv_results["category"],
        "sharpness": cv_results["sharpness"],
        "exposure": cv_results["exposure"],
        "issues": cv_results["issues"],
        "resolution": cv_results["resolution"],
        "file_size_kb": round(len(content) / 1024, 2),
        "created_at": datetime.utcnow()
    }

    result = await collection.insert_one(record)
    record["id"] = str(result.inserted_id)
    if "_id" in record:
        del record["_id"]

    return record

@app.get("/api/photos")
async def get_photos(category: Optional[str] = Query(None, pattern="^(good|review|poor)$")):
    query = {}
    if category:
        query["category"] = category

    cursor = collection.find(query).sort("created_at", -1)
    photos = []
    async for doc in cursor:
        photos.append(format_doc(doc))
    return photos

@app.patch("/api/photos/{photo_id}/category")
async def update_classification(photo_id: str, payload: ClassificationUpdate):
    if not ObjectId.is_valid(photo_id):
        raise HTTPException(status_code=400, detail="Invalid photo ID format.")

    result = await collection.update_one(
        {"_id": ObjectId(photo_id)},
        {"$set": {"category": payload.category}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Photo record not found.")

    return {"status": "success", "id": photo_id, "category": payload.category}