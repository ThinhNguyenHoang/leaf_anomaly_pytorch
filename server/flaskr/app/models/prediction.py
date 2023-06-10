import datetime
from app.extensions import db
from sqlalchemy import Column, Integer, DateTime
from dataclasses import dataclass

@dataclass
class Prediction(db.Model):
    id: int
    key_id: str
    input_files: list[str]
    results_files: list[str]

    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)
    key_id = db.Column(db.String(200))
    input_files = db.Column(db.Text)
    results_files = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(DateTime,default=datetime.datetime.utcnow )
    def __repr__(self):
        return f'<Prediction with key: "{self.key_id}">'
