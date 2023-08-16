from flask import Flask, render_template, request

from basic_text_summary import basic_summarizer
from NLP_text_summary import NLP_summarizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summary', methods = ['GET' , 'POST'])
def summary():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        if request.form['submit_button'] == 'basic_summary':
            summary, original_txt, _, _ = basic_summarizer(rawtext)
        
        
        elif request.form['submit_button'] == 'NLP_summary':
            summary, original_txt = NLP_summarizer(rawtext)
    return render_template('output.html', summary=summary, original_txt = original_txt)


if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run(debug = True)