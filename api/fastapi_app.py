from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from src.run_video import process_video
import os
import shutil
from pathlib import Path
import csv


app = FastAPI(
    title="Bird Detection & Weight Tracking API",
    description="Counts birds, tracks them with unique IDs, and estimates weight index from video",
    version="1.0"
)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "API Running Successfully"}


# --------------------------------------------------
# Health Check endpoint
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "OK"}


# --------------------------------------------------
# Main fixed video processing on input/input.mp4
# --------------------------------------------------
@app.get("/process-video")
def process_video_api():

    result = process_video()

    frame_data = result["frame_counts"]
    weight_data = result["weights"]
    unique_ids = list(weight_data.keys())

    formatted_response = {
        "video_name": "input/input.mp4",
        "total_unique_birds": len(unique_ids),
        "frame_count": len(frame_data),

        "unique_ids": unique_ids,

        "bird_details": [
            {"id": bird_id, "weight_index": round(weight_data[bird_id], 3)}
            for bird_id in unique_ids
        ],

        "birds_detected_each_frame": {
            f"frame_{frame}": count for frame, count in frame_data.items()
        }
    }

    return formatted_response


# --------------------------------------------------
# Video Download endpoint
# --------------------------------------------------
@app.get("/download-video")
def download_video():

    file_path = "output/output_video.mp4"

    if not os.path.exists(file_path):
        return {
            "error": "Video file not found. Run /process-video before downloading."
        }

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename="processed_birds.mp4"
    )


# --------------------------------------------------
# CSV Report Download endpoint
# --------------------------------------------------
@app.get("/download-csv")
def download_csv():

    result = process_video()

    csv_file = "output/report.csv"

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Bird_ID", "Average_Weight_Index"])
        for bid, w in result["weights"].items():
            writer.writerow([bid, w])

    return FileResponse(
        path=csv_file,
        media_type="text/csv",
        filename="bird_weight_report.csv"
    )


# --------------------------------------------------
# NEW → Analyze Uploaded Video Endpoint
# --------------------------------------------------
@app.post("/analyze_video")
async def analyze_video_api(file: UploadFile = File(...)):

    # File validation
    if file.content_type not in ["video/mp4", "video/mov", "video/avi"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    upload_path = Path("input/upload_video.mp4")

    # Save upload to disk
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run video processing
    result = process_video(
        video_path=str(upload_path),
        output_path="output/upload_processed.mp4"
    )

    frame_data = result["frame_counts"]
    weight_data = result["weights"]
    unique_ids = list(weight_data.keys())

    formatted_response = {
        "video_name": file.filename,
        "total_unique_birds": len(unique_ids),
        "frame_count": len(frame_data),

        "unique_ids": unique_ids,

        "bird_details": [
            {"id": bird_id, "weight_index": round(weight_data[bird_id], 3)}
            for bird_id in unique_ids
        ],

        "birds_detected_each_frame": {
            f"frame_{frame}": count for frame, count in frame_data.items()
        }
    }

    return formatted_response
