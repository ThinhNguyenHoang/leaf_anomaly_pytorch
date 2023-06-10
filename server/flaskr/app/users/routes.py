from flask import render_template, jsonify
import json
from app.models.user import User
from app.users import bp

from app.utils.json_encoder import AlchemyEncoder
@bp.route('/')
def index():
    users = User.query.order_by(User.id.desc()).all()
    # res = []
    # for user in users:
    #     json_str = json.dumps(user, cls=AlchemyEncoder)
    #     res.append(json_str)
    # print("RETUNING USERS:", res)
    return jsonify(users)

@bp.route('/<int:user_id>/')
def user(user_id):
    user = User.query.filter_by(id=user_id)
    return jsonify(user)