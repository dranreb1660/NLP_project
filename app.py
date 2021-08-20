import os
from chatbot_api import predict

# initialize the app
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']
        the_question = str(the_question)

        response = predict(the_question)
        return jsonify({"response": response})


# Heroku will set the port environment variable for
port = os.environ.get("PORT", 5100)
# set debug to false before deployment
app.run(debug=False, host="0.0.0.0", port=port)
