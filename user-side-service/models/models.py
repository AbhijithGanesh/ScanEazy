from .utils import db
from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship


class User(db.Model):
    __tablename__ = "user"
    username = db.Column(String(30), primary_key=True)
    email = db.Column(String(30))
    last_login = db.Column(DateTime(timezone=True), nullable=True)
    password = db.Column(String(40))
    submission = db.relationship("Submissions", back_populates="user")


class Submissions(db.Model):
    __tablename__ = "submission"
    id = db.Column(Integer(), primary_key=True)
    file_name = db.Column(String(40))
    unique_id = db.Column(UUID())
    user_id = db.Column(String, ForeignKey("user.username"), nullable=False)
    user = relationship("User", back_populates="submission")
