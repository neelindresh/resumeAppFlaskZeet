import flask
from data import details
app=flask.Flask(__name__)

@app.route("/")
def index():
    detailsforawards=details.AWARDS
    '''
    for i in detailsforawards:
        i["link"]=flask.url_for("static",filename=i["link"])
        i["url"]=flask.url_for("static",filename=i["url"])
        print(i)
    '''
    return flask.render_template("index-jinja.html",
                                 about_me=details.ABOUT_ME,
                                 about_details=details.ABOUT_DETAILS,
                                 skills=details.SKILLS,
                                 qualification=details.QUALIFICATION,
                                 projects=details.PROJECTS,
                                 awards=detailsforawards,
                                 roles_in_org=details.ROLES_AND_RESPOSIBILITY
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