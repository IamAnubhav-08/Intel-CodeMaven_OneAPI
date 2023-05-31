from flask import Flask, render_template, redirect, url_for
import pickle
import sys

from flask import request
# from sentence_preprocess import text_preprocess

app = Flask(__name__)

# model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def front():    
    return render_template('home.html')

@app.route("/input", methods=['POST'])
def handle_ajax_request():
     value = request.form.get('value')
     return redirect(url_for('predict', value=value))

@app.route("/predict", methods=["POST", "GET"])
def predict():
    input_text = request.form.values()
    model_num = request.args.get('value', None)
    if model_num==1:
        sent_type = 'Title'
    else:
        sent_type = 'Headline'

    # sent = text_preprocess(sent_type, input_text)
    prediction = 0
    # print(prediction)
    # if model_num==1:
    #  prediction = model_title.predict(input_text)
    # else:
    #  prediction = model_headline.predict(input_text)
    # if request.method=='POST':
     if(prediction>0):
        render_template('home.html',pred="{{Great! It is positive!}}")
     else:
        render_template('home.html',pred="OOPS! These words don't sound that good")
    

if __name__ == "__main__":
    app.run(debug=True, port=8000)
