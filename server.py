from flask import Flask,render_template, request


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
    print(request.args)
    return render_template("result.html")

app.run()