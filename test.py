import requests
import sys

if len(sys.argv) < 2:
    print("Usage: python test.py <path_to_your_image.jpg>")
    sys.exit(1)

image_path = sys.argv[1]

# Start the server if you haven't already:
# .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
url = "http://localhost:8000/api/v1/diagnose"

try:
    with open(image_path, "rb") as image_file:
        files = {"image": (image_path, image_file, "image/jpeg")}
        # Example environment data - change this to match your real location!
        data = {
            "environment": '{"region":"YourRegion","soil_type":"loam","season":"kharif","rainfall_last_7d_mm":0,"irrigation":"drip"}'
        }
        
        print(f"Sending {image_path} to the Advisory Engine...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- DIAGNOSIS RESULT ---")
            print(f"Crop/Disease: {result['diagnosis']['top']['crop']} - {result['diagnosis']['top']['disease']} (Confidence: {result['diagnosis']['top']['confidence']:.1%})")
            print(f"Damage Severity: {result['severity']['severity_pct']:.1f}% ({result['severity']['action_tier']})")
            print("\n--- TREATMENT PLAN SUMMARY ---")
            print(result['plan']['summary'])
        else:
            print(f"Error {response.status_code}: {response.text}")

except FileNotFoundError:
    print(f"Could not find the image at: {image_path}")
