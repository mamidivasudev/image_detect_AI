import os
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# Load the free open-source CLIP model once when the server starts
print("Loading free AI model into memory...")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# The HTML Frontend UI (embedded directly in the python script for simplicity)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citizen App Image Triage Tester</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background-color: #f5f7fa; color: #333; }
        .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        h2 { margin-top: 0; color: #1a202c; }
        .upload-zone { border: 2px dashed #cbd5e0; padding: 30px; text-align: center; border-radius: 8px; background: #f8fafc; cursor: pointer; transition: 0.2s; }
        .upload-zone:hover { border-color: #4299e1; background: #edf2f7; }
        #fileInput { display: none; }
        #preview { max-width: 100%; max-height: 300px; margin-top: 20px; border-radius: 6px; display: none; }
        .btn { display: inline-block; background: #3182ce; color: white; padding: 12px 24px; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 15px; width: 100%; font-size: 16px; }
        .btn:hover { background: #2b6cb0; }
        #loading { display: none; text-align: center; font-weight: bold; margin: 20px 0; color: #4a5568; }
        #result { margin-top: 25px; display: none; padding: 20px; border-radius: 8px; }
        .status-APPROVED { background: #c6f6d5; color: #22543d; border-left: 5px solid #38a169; }
        .status-REJECTED { background: #fed7d7; color: #742a2a; border-left: 5px solid #e53e3e; }
        .status-REVIEW { background: #feebc8; color: #744210; border-left: 5px solid #dd6b20; }
        .score-bar { background: #e2e8f0; height: 10px; border-radius: 5px; margin-top: 5px; overflow: hidden; }
        .score-fill { height: 100%; background: #4299e1; width: 0%; transition: width 0.5s ease-out; }
        .metric { margin-bottom: 15px; }
        .metric-title { font-weight: 600; display: flex; justify-content: space-between; }
    </style>
</head>
<body>

<div class="card">
    <h2>📸 Citizen App Image Triage Tester</h2>
    <p>Upload a picture to test if the local AI system correctly classifies it as valid infrastructure or a fake indoor/random upload.</p>
    
    <form id="uploadForm">
        <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <p>Drag & drop your image here or <strong>click to browse</strong></p>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            <img id="preview" alt="Upload Preview">
        </div>
        <button type="submit" class="btn">Analyze Image Accuracy</button>
    </form>

    <div id="loading">⏳ Processing image locally on your server...</div>

    <div id="result">
        <h3 id="decisionTitle" style="margin-top:0;">Decision: APPROVED</h3>
        <p id="decisionReason" style="margin-bottom:20px; font-size: 14px;"></p>
        
        <div class="metric">
            <div class="metric-title"><span>🛣️ Outdoor Road/Infrastructure Damage Score:</span> <span id="valRoad">0%</span></div>
            <div class="score-bar"><div id="barRoad" class="score-fill"></div></div>
        </div>
        <div class="metric">
            <div class="metric-title"><span>🏠 Indoor Home/Furniture Score:</span> <span id="valIndoor">0%</span></div>
            <div class="score-bar"><div id="barIndoor" class="score-fill" style="background:#e53e3e;"></div></div>
        </div>
        <div class="metric">
            <div class="metric-title"><span>❓ Random/Selfie/Irrelevant Score:</span> <span id="valRandom">0%</span></div>
            <div class="score-bar"><div id="barRandom" class="score-fill" style="background:#dd6b20;"></div></div>
        </div>
    </div>
</div>

<script>
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const preview = document.getElementById('preview');
        preview.src = reader.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
    document.getElementById('result').style.display = 'none';
}

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) return alert('Please select an image first!');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        
        document.getElementById('loading').style.display = 'none';
        
        // Show result box with background color based on status decision
        const resDiv = document.getElementById('result');
        resDiv.className = 'status-' + data.decision;
        resDiv.style.display = 'block';
        
        // Update Title & Text
        document.getElementById('decisionTitle').innerText = 'System Decision: ' + data.decision;
        document.getElementById('decisionReason').innerText = data.reason;
        
        // Update percentages text
        document.getElementById('valRoad').innerText = (data.scores.road * 100).toFixed(1) + '%';
        document.getElementById('valIndoor').innerText = (data.scores.indoor * 100).toFixed(1) + '%';
        document.getElementById('valRandom').innerText = (data.scores.random * 100).toFixed(1) + '%';
        
        // Update visual graphic bars
        document.getElementById('barRoad').style.width = (data.scores.road * 100) + '%';
        document.getElementById('barIndoor').style.width = (data.scores.indoor * 100) + '%';
        document.getElementById('barRandom').style.width = (data.scores.random * 100) + '%';
        
    } catch (err) {
        alert('Error processing image');
        document.getElementById('loading').style.display = 'none';
    }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the simple HTML interaction interface."""
    return HTML_CONTENT

@app.post("/predict")
async def predict_triage(file: UploadFile = File(...)):
    """Receives file upload, scores it using local CLIP model, returns data metrics."""
    # Open the uploaded citizen image
    image = Image.open(file.file).convert("RGB")

    # Business context label statements
    # text_queries = [
    #     "A photo of a public road, street, pothole, sidewalk, tree, or outdoor public infrastructure damage",
    #     "A photo of the inside of a residential house, living room, furniture, bedroom, or kitchen",
    #     "A photo of something random, a selfie, a document, or an abstract image"
    # ]
    text_queries = [
        # 1. The "Valid" Category 
        "A photo of outdoor public infrastructure damage, including a damaged road, potholes, cracked asphalt, broken pavement, a damaged bridge, waterlogging, broken drainage, or fallen trees on a street.",
        
        # 2. The "Fake/Indoor" Category
        "A photo of the inside of a house, an office, a bedroom, a kitchen, indoor furniture, or an indoor ceiling.",
        
        # 3. The "Fake/Random" Category
        "A close-up selfie, a person's face, a pet, food, a piece of paper, a text document, a receipt, a computer screen, or a completely black or blurry image."
    ]
   

    # Preprocess image and text inputs
    inputs = processor(text=text_queries, images=image, return_tensors="pt", padding=True)

    # Execute free inference locally
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).squeeze()

    # Extracted probability numbers
    scores = probs.tolist()
    road_score = scores[0]
    indoor_score = scores[1]
    random_score = scores[2]

    # Decide status outcome based on metrics rules
    if road_score > 0.70:
        decision = "APPROVED"
        reason = "Automated confirmation: Image matches outdoor road/infrastructure conditions."
    elif indoor_score > 0.60:
        decision = "REJECTED"
        reason = "Automated dismissal: High probability match for indoor home contents (fake upload)."
    else:
        decision = "REVIEW"
        reason = "Ambiguous validation context: Shifted to physical administrative verification queue."

    return {
        "decision": decision,
        "reason": reason,
        "scores": {
            "road": road_score,
            "indoor": indoor_score,
            "random": random_score
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Start web server on your local machine
    uvicorn.run(app, host="127.0.0.1", port=8000)
