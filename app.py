from flask import Flask, render_template, flash, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
import pandas as pd
import json
import plotly.express as px
import plotly
import os
from datetime import datetime
from textblob import TextBlob
import pyttsx3
import random
# Flask app setup
app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')
app.secret_key = "rutuja" 

from urllib.parse import quote_plus


# Set up MySQL database connection
password = quote_plus("root")  # Encode special characters
app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://root:{password}@localhost:3306/calmnest"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False  # To avoid warnings

db = SQLAlchemy(app)



# Login Manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Define the UserScore Model


# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    usn = db.Column(db.String(20), unique=True, nullable=False)
    pas = db.Column(db.String(1000), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Stress Analysis model
class StressAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stress_level = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Quiz and Game Data model
class QuizGameData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quiz_score = db.Column(db.Integer)
    game_score = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Load the trained model
model = pickle.load(open('stresslevel.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Signup route
@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == "POST":
        usn = request.form.get('usn')
        pas = request.form.get('pas')
        if not usn or not pas:
            flash("All fields are required!", "warning")
            return render_template("usersignup.html")
        encpassword = generate_password_hash(pas)
        existing_user = User.query.filter_by(usn=usn).first()
        if existing_user:
            flash("UserID is already taken", "warning")
            return render_template("usersignup.html")
        new_user = User(usn=usn, pas=encpassword)
        db.session.add(new_user)
        db.session.commit()
        flash("SignUp Successful! Please Login", "success")
        return redirect(url_for('login'))
    return render_template("usersignup.html")

# Login route
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == "POST":
        usn = request.form.get('usn')
        pas = request.form.get('pas')
        user = User.query.filter_by(usn=usn).first()
        if user and check_password_hash(user.pas, pas):
            login_user(user)
            flash("Login Successful", "info")
            return redirect(url_for('home'))
        else:
            flash("Invalid Credentials", "danger")
            return render_template("userlogin.html")
    return render_template("userlogin.html")

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logout Successful", "warning")
    return redirect(url_for('login'))

# Music route
@app.route('/music')
@login_required
def music():
    return render_template('music.html')

# Quiz and Game route
@app.route('/quizandgame')
@login_required
def quizandgame():
    return render_template('quizandgame.html')

# Exercises route
@app.route('/exercises')
@login_required
def exercises():
    return render_template('exercises.html')

# Quiz route
@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

# Game route
@app.route('/game')
def game():
    return render_template('game.html')

# Stress analysis route
@app.route('/analysis')
@login_required
def analysis():
    # Reading dataset
    train_df = pd.read_csv('dreaddit-train.csv', encoding='ISO-8859-1')
    train_df.drop(['text', 'post_id', 'sentence_range', 'id', 'social_timestamp'], axis=1, inplace=True)

    # Pie chart - Fixed here
    fig = px.pie(train_df, names='subreddit', title='Distribution of Subreddits')
    fig.update_traces(hovertemplate='%{label}: %{value}')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram of Stress Type
    train_df['label'].replace([0, 1], ['Not in Stress', 'In Stress'], inplace=True)
    fig2 = px.histogram(train_df, x="label", title='Distribution of Stress Type', color="label")
    fig2.update_layout(bargap=0.1)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Bar plot - Sentiment per subreddit
    fig3 = px.bar(train_df, x='subreddit', y='sentiment', title='Sentiment Score per Subreddit', color='subreddit')
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Scatter plot - Social Karma per subreddit
    fig4 = px.scatter(train_df, x='subreddit', y='social_karma', title='Social Karma per Subreddit', color='subreddit')
    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram - Confidence
    fig5 = px.histogram(train_df, x='confidence', marginal='box', title='Distribution of Confidence Scores')
    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram - Subreddit Distribution
    fig6 = px.histogram(train_df, x="subreddit", title='Distribution of Subreddits', color='subreddit')
    graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('analysis.html', graphJSON=graphJSON, graphJSON2=graphJSON2, graphJSON3=graphJSON3,
                           graphJSON4=graphJSON4, graphJSON5=graphJSON5, graphJSON6=graphJSON6)

# Stress detection form route
@app.route('/i')
def i():
    return render_template('stress.html')

@app.route('/stressdetect', methods=['POST'])
def stressdetect():
    try:
        # Retrieve input values safely
        sleeping_hours = request.form.get("rr", "").strip()
        blood_pressure = request.form.get("bp", "").strip()
        respiration_rate = request.form.get("bo", "").strip()
        heart_rate = request.form.get("hr", "").strip()

        # Validate inputs
        if not all([sleeping_hours, blood_pressure, respiration_rate, heart_rate]):
            flash("All fields are required!", "danger")
            return render_template('stress.html')

        # Convert inputs to integers
        try:
            sleeping_hours = int(sleeping_hours)
            blood_pressure = int(blood_pressure)
            respiration_rate = int(respiration_rate)
            heart_rate = int(heart_rate)
        except ValueError:
            flash("Please enter valid numbers!", "danger")
            return render_template('stress.html')

        # Rule-Based Stress Detection
        stress_level = "Normal Stress"
        message = "You are having Normal Stress!! Take Care of yourself."

        # Define thresholds for high stress
        if sleeping_hours < 5 or blood_pressure > 140 or respiration_rate > 500 or heart_rate > 120:
            stress_level = "High Stress"
            message = "You are having High Stress!! Consult a doctor and get the helpline number from our chatbot."

        return render_template('stress.html', prediction_text3=f'Stress Level is: {stress_level}. {message}')

    except Exception as e:
        flash(f"Error during detection: {str(e)}", "danger")
        return render_template('stress.html')

def detect_emotion(text):
    sentences = text.split('.')
    polarities = []

    for sentence in sentences:
        if sentence.strip():
            analysis = TextBlob(sentence)
            polarities.append(analysis.sentiment.polarity)

    if not polarities:
        return "No text provided.", ""

    avg_polarity = sum(polarities) / len(polarities)

    if avg_polarity > 0.5:
        emotion = "Very Happy 😄"
        quotes = [
            "Happiness is a choice, not a result.",
            "Start each day with a positive thought.",
            "Your joy is contagious — spread it!"
        ]
    elif avg_polarity > 0:
        emotion = "Happy 🙂"
        quotes = [
            "Keep your face always toward the sunshine.",
            "Small joys make big days.",
            "Happiness grows when shared."
        ]
    elif avg_polarity == 0:
        emotion = "Neutral 😐"
        quotes = [
            "Stay calm, be mindful, and enjoy the moment.",
            "Balance is the key to a peaceful mind.",
            "Even an ordinary day can be extraordinary."
        ]
    elif avg_polarity > -0.5:
        emotion = "Sad 😢"
        quotes = [
            "Storms make trees take deeper roots.",
            "It’s okay to feel down — better days are ahead.",
            "Every sunset brings the promise of a new dawn."
        ]
    else:
        emotion = "Very Sad 😭"
        quotes = [
            "You’ve made it through tough times before.",
            "Difficult roads often lead to beautiful destinations.",
            "Your current situation is not your final destination."
        ]

    quote = random.choice(quotes)
    return emotion, quote

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/emotion_detection', methods=['GET', 'POST'])
def emotion_detection():
    emotion = None
    quote = None
    if request.method == 'POST':
        user_text = request.form['user_text']
        emotion, quote = detect_emotion(user_text)

        # Speak the motivational quote on server machine
        speak_text(quote)

    return render_template('emotion_detection.html', emotion=emotion, quote=quote)

# Main function
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
