from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

#function that initializes the db and create the tables
def db_init(app):
    db.init_app(app)

    #Creates the table if the db doesn't exist
    with app.app_context():
        db.drop_all()
        db.create_all()