from flask import Flask,render_template, request
from score_model import scorer

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("page.html")
#Showing the result page
@app.route('/result')
def result():
    score = scorer(request.args)
    print(score)
    return render_template("result.html", score=score)

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)


