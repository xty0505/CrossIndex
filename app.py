from flask import Flask, g, current_app, render_template
from blueprint import crossindex

app = Flask(__name__)
app.register_blueprint(crossindex.bp)


@app.route('/')
def hello_world():
    content = 'Hello World'
    return render_template('index.html', content=content)


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
