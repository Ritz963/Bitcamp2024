from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

@app.route('/upload', methods=['POST'])
def upload_image():
    image = request.files['image']
    image.save("received_image.jpeg")
    return "Image received successfully", 200

if __name__ == '__main__':
    app.run(debug=True)
