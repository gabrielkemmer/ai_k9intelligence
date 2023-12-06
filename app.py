from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_session import Session
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash
import os
from transformers import AutoTokenizer, AutoModelForCausalLM  


app = Flask(__name__)
app.secret_key = 'Fn741953.741953'  # Change to a random secret key
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# MongoDB setup
client = pymongo.MongoClient("mongodb+srv://gabrielkemmer:Fn741953.741953@messages.dhykbwv.mongodb.net/?retryWrites=true&w=majority")
db = client["chatbot_database"]
users = db["users"]
chats = db["chats"]

path = os.getcwd()            
model_path = '/Users/gabrielkemmer/Library/CloudStorage/GoogleDrive-gabrielkemmer@k9agency.digital/My Drive/K9 Intelligence/Bot AI/fine_tuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path)


@app.route('/')
def initial():
    return render_template('initial.html')

@app.route('/set_email', methods=['POST'])
def set_email():
    email = request.form['email']
    session['email'] = email
    if chats.count_documents({"email": email}) > 0:
        return redirect(url_for('chat', email=email))
    else:
        return redirect(url_for('new_chat'))

    
@app.route('/new_chat')
def new_chat():
    if 'email' not in session:
        return redirect(url_for('initial'))
    return render_template('chat.html')

@app.route('/services')
def services():
    if 'email' not in session:
        return redirect(url_for('initial'))
    return render_template('services.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({"email": email})
        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']
            return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        users.insert_one({"email": email, "password": hashed_password})
        return redirect(url_for('login'))
    return render_template('register.html')


def generate_response(user_input):
    # Encode the user input and add end of string token
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate a response
    bot_output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the model output
    return tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)


@app.route('/message', methods=['POST'])
def message():
    if 'email' not in session:
        return jsonify({'reply': 'Session expired, please login again.'})

    user_message = request.form['message']
    reply = generate_response(user_message)

    # Save conversation to database
    chats.insert_one({"email": session['email'], "user_message": user_message, "bot_reply": reply})
    
    return jsonify({'reply': reply})


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))