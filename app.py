import os, glob
from flask import Flask, flash, request, redirect, render_template, send_file
from werkzeug.utils import secure_filename
import zipfile
import io
import pathlib

import cv2
import polygon_interacter as poly_i
from DocScanner import DocScanner

app=Flask(__name__)

# --------------- Setting Certain Configurations About the Code -------------- #
app.secret_key = "secret key"
widthImg = 1200
heightImg = 675
docCap = DocScanner(interactive = 0)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ----------------------------- Get current path ----------------------------- #
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# ---------------- Make upload directory if it does not exist ---------------- #
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------ Allowed extension you can set your own ------------------ #
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- Calculate Borders and Get Processed Data ----------------- #
def get_output():
    for image in os.listdir(UPLOAD_FOLDER):
        if image == '.DS_Store': continue
        img = cv2.imread(os.path.join(UPLOAD_FOLDER, image))

        # Finding Countours
        screenCnt = docCap.get_contour(img)

        # Interactive - NOT AVAILABLE IN WEB APP
        if docCap.interactive:
            screenCnt = docCap.interactive_get_contour(screenCnt, img)

        # Applying the perspective transformations
        warped = docCap.four_point_transform(img, screenCnt)

        # Preprocessing image for OCR
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sharpen1 = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen2 = cv2.addWeighted(gray, 1.5, sharpen1, -0.5, 0)

        cv2.imwrite(os.path.join(path, f'outputs/{image}'), sharpen2)

# -------------------------- Basic Homepage Template ------------------------- #
@app.route('/')
def upload_form():
    return render_template('index2.html')

# ------------------- Uploading Files to the Uploads Folder ------------------ #
@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'images' not in request.files:
            flash('No file part')
            return redirect(request.url)

        images = request.files.getlist('images')

        for image in images:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('File(s) successfully uploaded')
        get_output()
        return redirect('/')

# ---------------- Make the output files Available to Download --------------- #
@app.route('/download')
def images_zip():

    # Creating the Zip File of Outputs
    base_path = pathlib.Path('./outputs/')
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode = 'w') as z:
        for fileName in base_path.iterdir():
            z.write(fileName)
            os.unlink(fileName)
    data.seek(0)

    # Remove uploaded Files
    upload = pathlib.Path('./uploads/')
    for fileName in upload.iterdir():
        os.unlink(fileName)

    # Returning the zip file
    return send_file(data, mimetype='application/zip',
        as_attachment=True, attachment_filename='images.zip')


# --------------------- Run The App In Local Environment --------------------- #
if __name__ == "__main__":
    app.run(host = '127.0.0.1', port = 5000, debug = True, threaded = True)