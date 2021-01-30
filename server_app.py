from flask import Flask, request, jsonify, render_template
import pipelines
from utils import draw_dep_tree, preprocess

app = Flask(__name__)
nlp = pipelines.get_nlp()

@app.route('/pipe', methods=['POST'])
def pipe():
    input_str = request.form['input_str']
    paragraphs = preprocess(input_str)
    docs = [nlp(p) for p in paragraphs]
    
    return render_template('index.html', docs=docs, draw_dep_tree=draw_dep_tree)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# if __name__ == '__main__':
#       app.run(host='0.0.0.0', port=5000)