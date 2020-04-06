from flask import Flask,render_template, request
from score_model import scorer

app = Flask(__name__)


@app.route('/home')
def hello():
    return render_template("page.html")

@app.route('/')
def default():
    print(request.form)
    return "haha"

#Showing the result page
@app.route('/result')
def result():
    score = scorer(request.args)
    print(request.args)
    print(score)
    return render_template("result.html")

app.run()