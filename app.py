# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json['image']
    # Decode the base64 string
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    image.save("received_image.jpeg")  # Save or process the image as needed
    
    # Respond back to the frontend
    return jsonify({"message": "Image received successfully!"})
    pass

if __name__ == '__main__':
    app.run(debug=True)
