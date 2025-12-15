
import requests
import os

# Define the URL and the image path
url = 'http://localhost:8001/analyze'
image_path = r'd:/AIFace-final/AIFace/data/raw/face_shape_dataset/FaceShape Dataset/testing_set/Round/round (1).jpg'

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit(1)

# Prepare the payload
files = {'file': open(image_path, 'rb')}
data = {'gender': 'Female'}

try:
    print(f"Sending request to {url} with gender='Female'...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        json_response = response.json()
        print("\n--- Response ---")
        recs = json_response.get('recommendations', {})
        print(f"Beards: {recs.get('beards')}")
        print(f"Makeup: {recs.get('makeup')}")
        
        # Validation
        if recs.get('beards') == [] and len(recs.get('makeup', [])) > 0:
             print("\nSUCCESS: Logic appears correct (No beards, Has makeup)")
        else:
             print("\nFAILURE: Logic Incorrect (Beards found or Makeup missing)")
             
    else:
        print(f"Error: Status Code {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Exception: {e}")
