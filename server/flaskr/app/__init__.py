from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from config import Config

from app.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app = CORS(app)
    app.config.from_object(config_class)

    # Initialize Flask extensions here
    db.init_app(app)
    
    # Register blueprints here
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.users import bp as users_bp
    app.register_blueprint(users_bp, url_prefix='/users')

    @app.route('/test/')
    def test_page():
        return '<h1>Testing the Flask Application Factory Pattern</h1>'

    @app.route('/is_alive/')
    def is_alive():
        print("/isalive request")
        status_code = Response(status=200)
        return status_code
    # Predict route
    # @app.route("/predict", methods=["POST"])
    # def predict():
    #     print("Prediction::Processing")
    #     req_json = request.get_json()
    #     json_instances = req_json["instances"]
    #     X_list = [np.array(j["image"], dtype="uint8") for j in json_instances]
    #     X_transformed = torch.cat([transform(x).unsqueeze(dim=0) for x in X_list]).to(device)
    #     preds = model(X_transformed)
    #     preds_classes = [classes[i_max] for i_max in preds.argmax(1).tolist()]
    #     print(preds_classes)
    #     return jsonify({
    #         "predictions": preds_classes
    #     })
    return app