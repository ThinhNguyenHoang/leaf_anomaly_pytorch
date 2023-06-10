from app.extensions import db
from dataclasses import dataclass
from sqlalchemy.orm import Mapped, mapped_column
from app.models.prediction import Prediction

@dataclass
class User(db.Model):
    id: int
    username: str
    predictions: Mapped[Prediction]

    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150))
    password_hash = db.Column(db.Text)
    predictions = db.relationship(Prediction)

    def __repr__(self):
        return f'<User "{self.username}">'