import requests

def test_prediction(image_path):
    # URL of your Flask API
    url = 'http://localhost:5000/predict'
    
    # Prepare the image file
    files = {
        'file': open(image_path, 'rb')
    }
    
    # Make the request
    response = requests.post(url, files=files)
    
    # Print the result
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']['class']}")
        print(f"Confidence: {result['prediction']['confidence']}%")
    else:
        print(f"Error: {response.json()}")

# Test the API
test_prediction('Dataset/PotatoPlants/Potato___healthy/1f9870b3-899e-46fb-98c9-cfc2ce92895b___RS_HL 1816.JPG')