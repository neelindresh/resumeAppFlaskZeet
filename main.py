import flask
from data import details
app=flask.Flask(__name__)

@app.route("/")
def index():
    return flask.render_template("index-jinja.html",
                                 about_me=details.ABOUT_ME,
                                 about_details=details.ABOUT_DETAILS,
                                 skills=details.SKILLS,
                                 qualification=details.QUALIFICATION
                                 )
@app.route("/jinja/")
def index_jinja():
    return flask.render_template("index.html",
                                 about_me=details.ABOUT_ME,
                                 about_details=details.ABOUT_DETAILS,
                                 skills=details.SKILLS
                                 )

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5005)