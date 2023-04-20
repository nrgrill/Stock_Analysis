from flask import Flask, render_template, url_for, request
app = Flask(__name__)

@app.route("/", methods = ['POST', 'GET'])
def base():
    return render_template('base.html')

@app.route("/graph")
def graph():
    return render_template('graph.html')

if __name__ == '__main__':
    app.run(debug=True)