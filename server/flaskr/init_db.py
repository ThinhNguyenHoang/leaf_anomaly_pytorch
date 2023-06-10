from app import db, create_app
import json

from app.models.user import User
from app.models.prediction import Prediction

app = create_app()
with app.app_context():
    db.drop_all()
    db.create_all()
    user1 = User(username='asd', password_hash='123123')
    user2 = User(username='asd1', password_hash='123123')
    user3 = User(username='asd2', password_hash='123123')
    ADDRESS = 'https://img.freepik.com/free-vector/cute-rabbit-with-duck-working-laptop-cartoon-illustration_56104-471.jpg'
    dump = json.dumps([ADDRESS])
    prediction1 = Prediction(key_id='A',input_files=dump , results_files=dump, user_id=1)
    prediction2 = Prediction(key_id='b',input_files=dump , results_files=dump, user_id=1)
    prediction3 = Prediction(key_id='C',input_files=dump , results_files=dump, user_id=1)
    prediction4 = Prediction(key_id='D',input_files=dump , results_files=dump, user_id=1)


    db.session.add_all([user1, user2, user3])
    db.session.add_all([prediction1, prediction2, prediction3, prediction4])

    db.session.commit()