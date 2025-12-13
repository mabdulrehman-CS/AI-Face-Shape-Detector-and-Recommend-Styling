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
best_fine = os.path.join(project_root, 'models', 'checkpoints', 'best_fine.keras')
best_head = os.path.join(project_root, 'models', 'checkpoints', 'best_head.keras')

# Logic to load best available model
active_model_path = None
if os.path.exists(model_path):
    active_model_path = model_path
elif os.path.exists(best_fine):
    active_model_path = best_fine
elif os.path.exists(best_head):
    active_model_path = best_head

rules_path = os.path.join(project_root, 'src', 'recommendation', 'rules.json')

engine = None
if active_model_path and os.path.exists(active_model_path):
    print(f"Loading model from {active_model_path}")
    try:
        engine = RecommendationEngine(active_model_path, rules_path)
        print("Model Loaded Successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No model found. Inference will fail.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(os.path.join(current_dir, "static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {e}"

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
