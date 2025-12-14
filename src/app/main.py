import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np

# Force CPU for inference to avoid OOM with training process
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Step 1: Environment Setup Complete")

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(project_root, 'src'))

print(f"Step 2: Path added - {project_root}")

try:
    print("Step 3: Importing Engine...")
    print(" [INFO] Loading TensorFlow/Keras (This may take ~45-60 seconds on CPU)...")
    from recommendation.engine import RecommendationEngine
    print("Step 3: Engine Imported")
except Exception as e:
    print(f"CRITICAL ERROR IMPORTING ENGINE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Initialize Engine
model_path = os.path.join(project_root, 'models', 'final_model.keras')
best_fine_v2 = os.path.join(project_root, 'models', 'checkpoints', 'best_fine_v2.keras')
best_fine = os.path.join(project_root, 'models', 'checkpoints', 'best_fine.keras')
best_head = os.path.join(project_root, 'models', 'checkpoints', 'best_head.keras')

# Logic to load best available model
available_models = {
    "Fine-Tuned v2 (Epoch 30 - Latest)": best_fine_v2,
    "Fine-Tuned v1 (Epoch 20)": best_fine,
    "Base Model (Initial)": model_path,
    "Head Model (Stage 1)": best_head
}

active_model_path = None
# Prioritize the Latest Fine-Tuned Checkpoint (v2)
if os.path.exists(best_fine_v2):
    active_model_path = best_fine_v2
    print(f"Selecting Latest Fine-Tuned Model (v2): {best_fine_v2}")
elif os.path.exists(best_fine):
    active_model_path = best_fine
    print(f"Selecting Fine-Tuned Model (v1): {best_fine}")
elif os.path.exists(best_head):
    active_model_path = best_head
    print(f"Selecting Initial Head Model: {best_head}")
elif os.path.exists(model_path):
    active_model_path = model_path
    print(f"Selecting Base Final Model: {model_path}")

rules_path = os.path.join(project_root, 'src', 'recommendation', 'rules.json')

engine = None
def load_engine(path):
    global engine
    if path and os.path.exists(path):
        print(f"Loading model from {path}")
        try:
            engine = RecommendationEngine(path, rules_path)
            print("Model Loaded Successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"Model path not found: {path}")
        return False

# Initial Load
load_engine(active_model_path)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(os.path.join(current_dir, "static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {e}"

@app.get("/models")
async def get_models():
    # Return list of models that actually exist
    models = []
    current_model_name = "Unknown"
    
    for name, path in available_models.items():
        if os.path.exists(path):
            models.append({"name": name, "id": name})
            if engine and engine.model_path == path: 
                 current_model_name = name
                 
    return JSONResponse(content={"models": models, "active": current_model_name})

@app.get("/model_metrics")
async def get_metrics(model_id: str):
    # Mapping for metrics
    metrics_map = {
        "Fine-Tuned v2 (Epoch 30 - Latest)": "best_fine_v2.json",
        "Fine-Tuned v1 (Epoch 20)": "best_fine.json",
        "Base Model (Initial)": "final_model.json",
        "Head Model (Stage 1)": "best_head.json"
    }
    
    filename = metrics_map.get(model_id)
    if not filename:
        return JSONResponse(status_code=404, content={"error": "Metrics not found for this model"})
        
    metrics_path = os.path.join(project_root, 'models', 'metrics', filename)
    
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    else:
        return JSONResponse(status_code=404, content={"error": "Metrics file not found"})

from pydantic import BaseModel
class ModelSwitchRequest(BaseModel):
    model_id: str

@app.post("/switch_model")
async def switch_model(request: ModelSwitchRequest):
    target_path = available_models.get(request.model_id)
    if not target_path:
        return JSONResponse(status_code=400, content={"error": "Invalid model ID"})
    
    if not os.path.exists(target_path):
        return JSONResponse(status_code=404, content={"error": "Model file not found"})
        
    success = load_engine(target_path)
    if success:
        return JSONResponse(content={"status": "success", "message": f"Switched to {request.model_id}"})
    else:
        return JSONResponse(status_code=500, content={"error": "Failed to load model"})

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    if not engine:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    
    try:
        # Save temp file
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Predict
        result = engine.predict(temp_filename)
        
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        # Check for geometric/pipeline errors
        if "error" in result:
             return JSONResponse(status_code=400, content=result)
             
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    print("Starting Uvicorn Server on Port 8001...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        print(f"Uvicorn Failed: {e}")
