import os
from pathlib import Path
from api.constants import SCHEMA_DEFAULTS_PATH
from uuid import uuid4
from api.logger import logger
from json import loads
from flask import Flask, flash, request, redirect
from api.constants import ALLOWED_EXTENSIONS, UPLOAD_FOLDER, SECRET_KEY
from api.runner import process_file
from api.core.file import load_json


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = SECRET_KEY


@app.route("/image-api/")
def index():
    return "Hello World"


@app.route("/image-api/return-template")
def return_schema_template():
    return load_json(SCHEMA_DEFAULTS_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/image-api/upload-image", methods=["GET", "POST"])
def upload_file():
    _dirname = request.form.get("dirname")
    _filename = request.form.get("filename")
    _file = request.files.get("fileObj")
    if request.method == "POST":
        file = _file
        if file.filename == "":
            flash("No selected file")
            return 413  # I am a teapot
        if file:
            path = Path(app.config["UPLOAD_FOLDER"] + "/" + _dirname)
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
            file.save(os.path.join(path, f"{_filename}.png"))
            return {"status": "Saved"}
    return "GET is empty"


@app.route("/image-api/process-image", methods=["POST", "GET"])
def process_image():

    if request.method == "POST":
        args: dict = {"autoAlign": True}
        json_body: dict = loads(request.data.decode())
        dir_name = json_body.get("dirname")
        file_path = (
            app.config.get("UPLOAD_FOLDER")
            + "/"
            + dir_name
            + "/"
            + json_body.get("filename")
        )
        data = json_body.get("schema")
        logger.critical(data)
        try:
            return process_file(
                file_path,
                data,
                args,
            )[0]
        except Exception:
            return "Invalid Schema"

    return "GET is empty"
