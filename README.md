# bird_detection_weight
Bird Detection & Weight Estimation System

This project detects birds in video, assigns unique tracking IDs, counts them across frames, and estimates relative weight index using bounding box area. A FastAPI backend exposes endpoints to analyze videos and download processed results.

1. Features

YOLO-based bird detection

Unique ID tracking per bird

Weight indexing using bounding box area

JSON output for analytics

Download processed annotated video

Upload any new video for analysis

Download CSV weight summary report


2. Folder Structure
Poultry project/
│
├── api/                     → FastAPI application
│   └── fastapi_app.py
│
├── src/
│   ├── run_video.py         → main tracking + logic
│   └── weight_utils.py      → weight estimation logic
│
├── input/                   → original video storage
│   └── input.mp4
│
├── output/
│   ├── output_video.mp4     → processed annotated video
│   ├── bird_weight_report.csv
│   ├── sample_output.json   → JSON output sample (required)
│
├── README.md
└── requirements.txt


3. Environment Setup
Create virtual environment:
python -m venv my_env

Activate environment:

Windows

my_env\Scripts\activate

Install required packages:
pip install -r requirements.txt


4. Run the API server

Inside project root:

uvicorn api.fastapi_app:app --reload


API will run at:

http://127.0.0.1:8000

Documentation UI:

http://127.0.0.1:8000/docs


5. API Endpoints
Root test
GET /

Health check
GET /health

Analyze default included video
GET /process-video

Upload and analyze new video
POST /analyze_video

Download processed video
GET /download-video

Download weight CSV
GET /download-csv


6. CURL Example for /analyze_video

Use this to send a video to backend for analysis:

curl -X POST "http://127.0.0.1:8000/analyze_video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@<video_path>.mp4"


Example:

file=@input/input.mp4


7. Method Explanation (Implementation Details)
Bird counting approach:

YOLOv8n detector identifies bounding boxes (class 14 = bird)

Model runs at sample FPS to reduce compute load

Unique ID assignment using persistent track IDs

Final unique bird count = number of distinct IDs

Weight estimation logic:

Weight index computed using bounding box pixel area

Larger birds → higher weight index range

Weight index averaged across all frames per bird

Output returned as weight_index per ID


8. Demo Output Files

Available inside output folder:

output_video.mp4 → fully annotated bird tracking video

sample_output.json → JSON returned from /analyze_video

bird_weight_report.csv → summary table


9. Notes

All detections run on CPU

Weight index is estimation-based, not actual kilograms

Any mp4 video can be analyzed



