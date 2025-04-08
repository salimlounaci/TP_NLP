from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

DB_URI = "sqlite:///your_database.db"

DB = SQLAlchemy()

def create_app(config):
    app = Flask(__name__)

    if "SQLALCHEMY_DATABASE_URI" not in config:
        config["SQLALCHEMY_DATABASE_URI"] = DB_URI

    app.config.update(config)

    DB.init_app(app)

    with app.app_context():
        from api.models import SaleWeeklyRaw
        DB.create_all()

    @app.route('/post_sales', methods=['POST'])
    def post_sales():
        data = request.json
        for row in data:
            new_entry = SaleWeeklyRaw(
                year_week=row['year_week'],
                vegetable=row['vegetable'],
                sales=row['sales']
               )
        DB.session.add(new_entry)
        DB.session.commit()

        return jsonify({"status": "success"}), 200

    @app.route('/get_raw_data', methods=['GET'])
    def get_raw_data():
        entries = DB.session.query(SaleWeeklyRaw).filter_by(
            year_week=202001,
            vegetable="babar",
        )

        return jsonify([row.json() for row in entries]), 200

    return app
