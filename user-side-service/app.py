from json import dumps, loads
from uuid import uuid4

import requests as _requ
from flask import Flask, request, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_required, logout_user
from werkzeug.datastructures import Headers

from constants import (ALLOWED_EXTENSIONS, CONNECTION_STRING, SECRET_KEY,
                       UPLOAD_FOLDER, UPLOAD_SERVICE_ENDPOINT)
from logger import logger
from models.models import Submissions
from models.models import User as base
from models.utils import db

app = Flask(__name__)
login_manager = LoginManager()
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config.update({"SQLALCHEMY_DATABASE_URI": f"{CONNECTION_STRING}"})
app.config.update({"SECRET_KEY": SECRET_KEY})
app.config.update({"UPDATE_FOLDER": f"{UPLOAD_FOLDER}"})

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

@login_required
@app.route("/create-submission", methods=["POST"])
def create_submission():
    try:
        _fileObj = request.files.get("file")
        assigned_unique_id: str = str(uuid4())
        submission = Submissions(
            user_id=session.get("username"),
            file_name=_fileObj.filename,
            unique_id=f"{assigned_unique_id}",
        )
        if "file" not in request.files:
            return "No file part"

        if _fileObj.filename == "":
            return 413  # I am a teapot

        if validate_files(_fileObj.filename):
            _req = _requ.post(
                UPLOAD_SERVICE_ENDPOINT,
                files={"fileObj": _fileObj},
                data={"dirname": session["username"],
                      "filename": submission.unique_id},
            )
            logger.info(_fileObj.filename)
            logger.info(_req.text)
            db.session.add(submission)
            db.session.commit()
            return {"fileSaved": "Submission got saved", "fileId": assigned_unique_id+".png"}
        else:
            return "Give a proper image file"

    except Exception as err:
        logger.error(f"Something broke at :{err}")
        return {"message": 400}


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
@app.route("/validate-result", methods=["POST"])
def validate_an_image():
    user = session.get("username")
    # logger.critical(request.json)
    json_obj = (request.json)
    process_image_url = UPLOAD_SERVICE_ENDPOINT.replace(
        "/upload-image", "/process-image"
    )
    # logger.critical(loads(request.data.decode()))
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
    return (_res.text)


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
    logout_user()
    return session.get("username")
