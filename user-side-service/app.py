import datetime
import os
import requests as _requ
from pathlib import Path
from json import loads, dumps
import time
from uuid import uuid4

from constants import (
    ALLOWED_EXTENSIONS,
    CONNECTION_STRING,
    SECRET_KEY,
    UPLOAD_FOLDER,
    UPLOAD_SERVICE,
    UPLOAD_SERVICE_ENDPOINT,
)
from flask import Flask, jsonify, redirect, request, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_required, logout_user
from logger import logger
from models.models import Submissions,User as base
from models.utils import db
from werkzeug.datastructures import Headers

app = Flask(__name__)
login_manager = LoginManager()
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config.update({"SQLALCHEMY_DATABASE_URI": f"{CONNECTION_STRING}"})
app.config.update({"SECRET_KEY": SECRET_KEY})
app.config.update({"UPDATE_FOLDER": f"{UPLOAD_FOLDER}"})

time.sleep(5)
db.init_app(app)
login_manager.init_app(app)

with app.app_context():
    db.create_all()
logger.info("App is running!")


class UserMixedIn(UserMixin, base):
    pass


@login_manager.user_loader
def user_loader(user_id):
    return UserMixedIn.query.get(user_id)


def validate_files(filename: str) -> bool:
    data = filename.split(".")[-1]
    if data in ALLOWED_EXTENSIONS:
        return True
    else:
        return False


@app.route("/users", methods=["POST"])
def create_user():
    data = request.json
    head = Headers()
    try:
        user = base(
            username=data.get("name"),
            email=data.get("email"),
            password=data.get("password"),
        )

    except Exception as e:
        logger.error(f"Something broke at :{e}")
        return 400
    db.session.add(user)
    db.session.commit()
    head.add("User", data.get("name"))
    return "User got saved"


def accepted(filename) -> bool:
    data = filename.split(".")[-1]
    if data in ALLOWED_EXTENSIONS:
        return True
    else:
        return False


@login_required
@app.route("/create-submission", methods=["POST"])
def create_submission():
    try:
        _fileObj = request.files.get("file")
        submission = Submissions(
            user_id=session.get("username"),
            file_name=_fileObj.filename,
            unique_id=f"{uuid4()}",
        )
        if "file" not in request.files:
            return "No file part"

        if _fileObj.filename == "":
            return 413  # I am a teapot

        if validate_files(_fileObj.filename):
            _req = _requ.post(
                UPLOAD_SERVICE_ENDPOINT,
                files={"fileObj": _fileObj},
                data={"dirname": session["username"], "filename": submission.unique_id},
            )
            logger.info(_fileObj.filename)
            logger.info(_req.text)
            db.session.add(submission)
            db.session.commit()
            return "Submission got saved"
        else:
            return "Give a proper image file"

    except Exception as err:
        logger.error(f"Something broke at :{err}")
        return {"message":400}


@login_required
@app.route("/get-all-submissions", methods=["GET"])
def get_all_submissions():
    user = session.get("username")
    _data = Submissions.query.filter_by(user_id=user)
    data = []
    for i in _data:
        data.append([i.unique_id, i.file_name])
    logger.critical(data)
    return data


@login_required
@app.route("/validate-result")
def validate_an_image():
    user = session.get("username")
    json_obj = loads(request.data.decode())
    process_image_url = UPLOAD_SERVICE_ENDPOINT.replace(
        "/upload-image", "/process-image"
    )
    logger.critical(request.data.decode())
    _res = _requ.post(
        process_image_url,
        data=dumps(
            {
                "schema": json_obj.get("schema"),
                "dirname": user,
                "filename": json_obj.get("filename"),
            }
        ),
    )
    return loads(_res.text)


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = UserMixedIn.query.filter_by(
        username=data.get("username"), password=data.get("password")
    ).first()
    if user is not None:
        session["logged_in"] = True
        session["username"] = data.get("username")
        return "Authenticated"
    else:
        logger.info("Authentication failure")
        return "Failure"


@login_required
@app.route("/logout", methods=["POST"])
def logout():
    return session.get("username")
