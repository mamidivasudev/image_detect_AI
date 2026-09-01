import os
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Load free open-source AI models (CLIP for Triage + BLIP for Captioning) once when server starts
print("Loading free AI models into memory...")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
print("AI models loaded successfully!")

CATEGORY_DESCRIPTIONS = {
    "Bridge damaged":    "A photo of a cracked, broken, or structurally damaged bridge over water or road.",
    "Parapet damaged":   "A photo of a broken or missing parapet wall or railing on a bridge.",
    "Degraded road":     "A photo of a severely degraded, worn out, crumbling or damaged road surface.",
    "Railing damaged":   "A photo of a bent, broken, or missing metal railing or guardrail on a road.",
    "Potholes":          "A photo of potholes or deep holes in the surface of a road.",
    "Structure damaged": "A photo of a broken concrete retaining wall, cracked roadside brick boundary, or damaged concrete drainage culvert. A concrete wall structure is broken.",
    "Fallen tree":       "A photo of a fallen tree blocking or lying on a road or highway.",
    "Road overtopped":   "A photo of a road flooded or overtopped by water, river, or rain.",
    "Bump not painted":  "A photo of an unpainted, invisible, or poorly marked speed bump on a road.",
    "Breach on road":    "A photo of a road where a massive gap or hole has formed because the asphalt has collapsed or washed away. The road surface itself is missing."
}

# The HTML Frontend UI (embedded directly in the python script for simplicity)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citizen App Image Triage Tester</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px auto; padding: 0 20px; background-color: #f5f7fa; color: #333; overflow-y: hidden; }
        .header { margin-bottom: 15px; text-align: center; }
        h2 { margin-top: 0; margin-bottom: 5px; color: #1a202c; font-size: 22px; }
        p.subtitle { margin-top: 0; font-size: 14px; color: #4a5568; margin-bottom: 10px; }
        .split-layout { display: flex; gap: 20px; align-items: flex-start; max-width: 1100px; margin: 0 auto; height: 85vh; }
        .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); width: 50%; box-sizing: border-box; }
        .upload-zone { border: 2px dashed #cbd5e0; padding: 15px; text-align: center; border-radius: 8px; background: #f8fafc; cursor: pointer; transition: 0.2s; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 260px; }
        .upload-zone:hover { border-color: #4299e1; background: #edf2f7; }
        #fileInput { display: none; }
        #preview { max-width: 100%; max-height: 200px; margin-top: 10px; border-radius: 6px; display: none; object-fit: contain; }
        #uploadPrompt { margin: 0; font-size: 14px; }
        #loading { display: none; text-align: center; font-weight: bold; margin: 0; color: #4a5568; padding: 20px; background: white; border-radius: 12px; }
        #result { display: none; padding: 20px; border-radius: 12px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .description-box { background: rgba(255, 255, 255, 0.6); border-left: 4px solid #3182ce; padding: 10px 12px; border-radius: 6px; margin-bottom: 15px; font-size: 13px; color: #2d3748; }
        .status-APPROVED { background: #c6f6d5 !important; color: #22543d; border-left: 5px solid #38a169; }
        .status-REJECTED { background: #fed7d7 !important; color: #742a2a; border-left: 5px solid #e53e3e; }
        .status-REVIEW { background: #feebc8 !important; color: #744210; border-left: 5px solid #dd6b20; }
        .score-bar { background: #e2e8f0; height: 8px; border-radius: 4px; margin-top: 4px; overflow: hidden; }
        .score-fill { height: 100%; background: #4299e1; width: 0%; transition: width 0.5s ease-out; }
        .metric { margin-bottom: 12px; }
        .metric-title { font-weight: 600; display: flex; justify-content: space-between; font-size: 13px;}
        select { padding: 8px 12px; width: 100%; margin-top: 5px; border-radius: 6px; border: 1px solid #cbd5e0; font-size: 14px; background: #fff; cursor: pointer; }
        label { font-size: 14px; }
    </style>
</head>
<body>

<div class="header">
    <h2>📸 Image Triage Tester</h2>
</div>

<div class="split-layout">
    <!-- LEFT SIDE: Upload & Category -->
    <div class="card">
        <form id="uploadForm">
            <div style="margin-bottom: 15px;">
                <label for="categorySelect"><strong>Select Complaint Category:</strong></label>
                <select id="categorySelect" onchange="triggerAnalysisIfReady()">
                    <option value="Bridge damaged">Bridge damaged</option>
                    <option value="Parapet damaged">Parapet damaged</option>
                    <option value="Degraded road">Degraded road</option>
                    <option value="Railing damaged">Railing damaged</option>
                    <option value="Potholes">Potholes</option>
                    <option value="Structure damaged">Structure damaged</option>
                    <option value="Fallen tree">Fallen tree</option>
                    <option value="Road overtopped">Road overtopped</option>
                    <option value="Bump not painted">Bump not painted</option>
                    <option value="Breach on road">Breach on road</option>
                </select>
            </div>
            
            <label><strong>Upload Image:</strong></label>
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()" style="margin-top: 5px;">
                <p id="uploadPrompt">Drag & drop your image here or <strong>click to browse</strong></p>
                <input type="file" id="fileInput" accept="image/*" onchange="previewImageAndAnalyze(event)">
                <img id="preview" alt="Upload Preview">
            </div>
        </form>
    </div>

    <!-- RIGHT SIDE: Analysis Results -->
    <div style="width: 50%;">
        <div id="loading" class="card" style="width: 100%;">
            ⏳ Processing image locally on your server...
        </div>

        <div id="result" class="card" style="width: 100%;">
            <h3 id="decisionTitle" style="margin-top:0; margin-bottom: 10px; font-size: 18px;">Decision: APPROVED</h3>
            <p id="decisionReason" style="margin-top:0; margin-bottom:12px; font-size: 13px; line-height: 1.4;"></p>
            
            <div id="descriptionBox" class="description-box" style="display:none;">
                <strong>🖼️ AI Image Description:</strong> <span id="imageDescText"></span>
            </div>
            
            <div class="metric">
                <div class="metric-title"><span id="lblSelectedCategory">🎯 Selected Category Match:</span> <span id="valRoad">0%</span></div>
                <div class="score-bar"><div id="barRoad" class="score-fill"></div></div>
            </div>
            <div class="metric" id="bestMatchMetric" style="display:none;">
                <div class="metric-title"><span id="lblBestCategory">🏆 Top Matching Damage Category:</span> <span id="valBest">0%</span></div>
                <div class="score-bar"><div id="barBest" class="score-fill" style="background:#38a169;"></div></div>
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
        
        <!-- Placeholder when nothing is uploaded yet -->
        <div id="placeholder" class="card" style="width: 100%; text-align: center; color: #a0aec0; padding: 40px 20px;">
            <div style="font-size: 32px; margin-bottom: 10px;">🔍</div>
            <h3 style="margin: 0 0 5px 0;">Awaiting Image</h3>
            <p style="margin: 0; font-size: 13px;">Select a category and upload an image.</p>
        </div>
    </div>
</div>

<script>
let currentFile = null;

function previewImageAndAnalyze(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    currentFile = file;
    
    const reader = new FileReader();
    reader.onload = function(){
        const preview = document.getElementById('preview');
        preview.src = reader.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    // Auto trigger analysis
    runAnalysis();
}

function triggerAnalysisIfReady() {
    if (currentFile) {
        runAnalysis();
    }
}

async function runAnalysis() {
    if (!currentFile) return;

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('category_name', document.getElementById('categorySelect').value);

    // Update UI states
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('result').style.display = 'none';
    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        
        document.getElementById('loading').style.display = 'none';
        
        const resDiv = document.getElementById('result');
        resDiv.className = 'card status-' + data.decision;
        resDiv.style.display = 'block';
        
        document.getElementById('decisionTitle').innerText = 'System Decision: ' + data.decision;
        document.getElementById('decisionReason').innerText = data.reason;
        
        if (data.description) {
            document.getElementById('imageDescText').innerText = data.description;
            document.getElementById('descriptionBox').style.display = 'block';
        } else {
            document.getElementById('descriptionBox').style.display = 'none';
        }
        
        const catName = document.getElementById('categorySelect').value;
        document.getElementById('lblSelectedCategory').innerText = "🎯 Selected Category ('" + catName + "') Score:";
        document.getElementById('valRoad').innerText = data.scores.road.toFixed(1) + '%';
        document.getElementById('barRoad').style.width = data.scores.road + '%';
        
        if (data.best_match && data.best_match !== catName && data.all_scores && data.all_scores[data.best_match]) {
            const bestScore = data.all_scores[data.best_match];
            document.getElementById('lblBestCategory').innerText = "🏆 Top Match ('" + data.best_match + "') Score:";
            document.getElementById('valBest').innerText = bestScore.toFixed(1) + '%';
            document.getElementById('barBest').style.width = bestScore + '%';
            document.getElementById('bestMatchMetric').style.display = 'block';
        } else {
            document.getElementById('bestMatchMetric').style.display = 'none';
        }
        
        document.getElementById('valIndoor').innerText = data.scores.indoor.toFixed(1) + '%';
        document.getElementById('valRandom').innerText = data.scores.random.toFixed(1) + '%';
        
        document.getElementById('barIndoor').style.width = data.scores.indoor + '%';
        document.getElementById('barRandom').style.width = data.scores.random + '%';
        
    } catch (err) {
        alert('Error processing image');
        document.getElementById('loading').style.display = 'none';
        document.getElementById('placeholder').style.display = 'block';
    }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the simple HTML interaction interface."""
    return HTMLResponse(content=HTML_CONTENT, status_code=200, media_type="text/html; charset=utf-8")

@app.post("/predict")
@app.post("/check_image_status")
async def predict_triage(file: UploadFile = File(...), category_name: str = Form("DEGREDED ROADS")):
    """Receives file upload, scores it using local CLIP model, returns data metrics."""
    image = Image.open(file.file).convert("RGB")

    # ✅ STEP 0: Generate AI natural language description using Salesforce BLIP
    caption_inputs = caption_processor(image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = caption_model.generate(**caption_inputs, max_new_tokens=50)
        image_description = caption_processor.decode(caption_ids[0], skip_special_tokens=True).capitalize()

    # ✅ STEP 1: Check against ALL 11 damage categories + indoor + random + normal road
    # This lets us find WHICH damage type the image best matches
    all_queries = list(CATEGORY_DESCRIPTIONS.values()) + [
        "A photo taken inside a house, bedroom, kitchen, office, or any indoor space with furniture, appliances, or household items.",
        "A selfie, portrait, person's face, pet animal, food item, clothing, paper document, receipt, toys, tools, cups, bottles, close-up of electronics, keyboard, laptop, phone, or computer screen.",
        "A photo of a clean, smooth, well-maintained, undamaged road or highway in perfect condition with no damage.",
        "A very blurry, out of focus, shaky, or low quality image where it is hard to see details clearly."
    ]

    all_labels = list(CATEGORY_DESCRIPTIONS.keys()) + ["INDOOR", "RANDOM", "NORMAL_ROAD", "BLURRY"]

    inputs = processor(text=all_queries, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze()

    scores = probs.tolist()

    # Map label → score
    score_map = {label: round(scores[i] * 100, 1) for i, label in enumerate(all_labels)}

    selected_score  = score_map.get(category_name, 0)
    indoor_score    = score_map.get("INDOOR", 0)
    random_score    = score_map.get("RANDOM", 0)
    normal_road_score = score_map.get("NORMAL_ROAD", 0)
    blurry_score    = score_map.get("BLURRY", 0)

    # Best matching damage category (excluding indoor/random/normal_road/blurry)
    road_scores = {k: v for k, v in score_map.items() if k not in ["INDOOR", "RANDOM", "NORMAL_ROAD", "BLURRY"]}
    best_match_label = max(road_scores, key=road_scores.get)
    best_match_score = road_scores[best_match_label]

    # ✅ STEP 2: Decision logic
    if (indoor_score + random_score) > best_match_score and (indoor_score + random_score) > 10:
        # Looks more like indoor/random than any road damage
        decision = "REJECTED"
        reason = f"Image appears to be indoor or irrelevant. Not a valid road damage photo."
        
    elif indoor_score > 30 or random_score > 30:
        # Clearly fake/indoor/random image
        decision = "REJECTED"
        reason = f"Image appears to be indoor or irrelevant. Not a valid road damage photo."

    elif blurry_score > 10:
        # Image is too blurry
        decision = "REVIEW"
        reason = f"Image is too blurry or out of focus (blurry score: {blurry_score}%). Needs manual review."

    elif normal_road_score > selected_score:
        # Image looks like a normal undamaged road — not actually damaged
        decision = "REJECTED"
        reason = f"Image appears to be a normal undamaged road (normal road score: {normal_road_score}%). No visible damage matching '{category_name}'."

    elif best_match_label == category_name and selected_score > 15:
        # Image best matches the selected category ✅
        decision = "APPROVED"
        reason = f"Image correctly matches '{category_name}' ({selected_score}% confidence)."

    elif best_match_score > 20 and best_match_label != category_name:
        # Image is a road image BUT matches a DIFFERENT damage category
        decision = "REJECTED"
        reason = f"Image looks like '{best_match_label}' ({best_match_score}%), not '{category_name}'. Please upload correct image."

    else:
        # Unclear - send for manual review
        decision = "REVIEW"
        reason = f"Image is unclear. Best match is '{best_match_label}' ({best_match_score}%). Needs manual review."

    return {
        "decision": decision,
        "reason": reason,
        "description": image_description,
        "best_match": best_match_label,
        "scores": {
            "road":   round(selected_score, 1),
            "indoor": round(indoor_score, 1),
            "random": round(random_score, 1)
        },
        "all_scores": score_map
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
