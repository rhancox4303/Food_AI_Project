import os
from os.path import join, dirname
import nutrition_data
import model  # Import the python file containing the ML model
from flask import Flask, request, render_template, redirect  # Import flask libraries
from werkzeug.utils import secure_filename

# Define Flask App.
app = Flask(__name__, template_folder="templates", static_folder="static")
# Define static uploads path.
UPLOADS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/static/uploads'

# Default GET route. Returns index.html
@app.route('/')
def index():
    return render_template('index.html')

# POST Route. Handles image upload.
@app.route('/', methods=['POST'])
def upload_image():
    # If image is not uploaded then redirect back to index.html
    if "image" not in request.files:
        return redirect(request.url)

    # Get image file and file name.
    file = request.files['image']
    fileName = secure_filename(file.filename)

    # Define the file path and save the image to that path.
    filePath = os.path.join(UPLOADS_PATH, fileName)
    file.save(filePath)

    # Predict the image given the filePath.
    modelOutput, predicted_class = model.predict_image(filePath)

    # Get the url to the uploaded image.
    url = request.base_url + 'static/uploads/'

    # Re-render index.html with the file path, modelOutput, and relevant nutrition data passed to the front end.
    return render_template('index.html', filePath=url + fileName, modelOutput=modelOutput, nutrition_data=
    nutrition_data.nutrition_data[nutrition_data.get_index()[predicted_class]])
