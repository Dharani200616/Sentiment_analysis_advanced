import os
import pickle
import sqlite3
import base64
from flask import Flask, request, jsonify, session, send_from_directory, redirect, url_for
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask import render_template
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
CORS(app, supports_credentials=True)
bcrypt = Bcrypt(app)

# ---------- Database Setup ----------
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  sentiment TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# ---------- Load Sentiment Model ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    
    if hasattr(loaded_model, 'named_steps'):
        # Loaded a scikit-learn Pipeline
        vectorizer = loaded_model.named_steps['vec']
        classifier = loaded_model.named_model = loaded_model.named_steps['clf']
        # Wait, let's make sure it's just:
        vectorizer = loaded_model.named_steps['vec']
        classifier = loaded_model.named_steps['clf']
        print("[INFO] Successfully loaded model from scikit-learn Pipeline.")
    else:
        # Loaded a tuple (vectorizer, classifier)
        vectorizer, classifier = loaded_model
        print("[INFO] Successfully loaded model from legacy tuple format.")
except Exception as e:
    print(f"[WARNING] Error loading model.pkl: {e}. Trying joblib load or fallback...")
    try:
        import joblib
        loaded_model = joblib.load(MODEL_PATH)
        if hasattr(loaded_model, 'named_steps'):
            vectorizer = loaded_model.named_steps['vec']
            classifier = loaded_model.named_steps['clf']
        else:
            vectorizer, classifier = loaded_model
        print("[INFO] Successfully loaded model via joblib.")
    except Exception as e2:
        print(f"[ERROR] Deserialization failed: {e2}. Creating a dummy vectorizer/classifier to prevent app crash.")
        # Fallback to prevent crash
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vectorizer = TfidfVectorizer()
        classifier = LogisticRegression()
        # Fit dummy data
        vectorizer.fit(["dummy text"])
        classifier.fit(vectorizer.transform(["dummy text"]), [1])

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'images')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ---------- Routes ----------

@app.route('/')
@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


@app.route('/dashboard', methods=['GET'])
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/graphs', methods=['GET'])
@app.route('/analytics', methods=['GET'])
def graphs_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('graphs.html')


@app.route('/images/<filename>')
def serve_image(filename):
    """Serve generated report images."""
    return send_from_directory(IMG_DIR, filename)


@app.route('/chart-images', methods=['GET'])
def chart_images():
    """Return base64-encoded chart images for the dashboard."""
    charts = {}
    for name in ['confusion_matrix', 'feature_importance',
                 'sentiment_distribution', 'model_accuracy',
                 'roc_curve', 'pr_curve']:
        path = os.path.join(IMG_DIR, f'{name}.png')
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                charts[name] = 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()
        else:
            charts[name] = None
    return jsonify(charts)


@app.route('/stats', methods=['GET'])
def stats():
    """Return aggregate stats for all users (public) and current user if logged in."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM history")
    total_analyses = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM history WHERE sentiment='Positive'")
    total_pos = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    c.execute("SELECT AVG(confidence) FROM history")
    avg_conf = c.fetchone()[0] or 0.0

    user_stats = {}
    if 'user_id' in session:
        uid = session['user_id']
        c.execute("SELECT COUNT(*) FROM history WHERE user_id=?", (uid,))
        user_stats['total'] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM history WHERE user_id=? AND sentiment='Positive'", (uid,))
        user_stats['positive'] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM history WHERE user_id=? AND sentiment='Negative'", (uid,))
        user_stats['negative'] = c.fetchone()[0]
        c.execute("SELECT AVG(confidence) FROM history WHERE user_id=?", (uid,))
        user_stats['avg_confidence'] = round((c.fetchone()[0] or 0.0) * 100, 1)

    conn.close()
    return jsonify({
        'total_analyses': total_analyses,
        'total_positive': total_pos,
        'total_negative': total_analyses - total_pos,
        'total_users': total_users,
        'avg_confidence': round(avg_conf * 100, 1),
        'user': user_stats
    })


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        return jsonify({'message': 'User created successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.check_password_hash(user[1], password):
        session['user_id'] = user[0]
        session['username'] = username
        return jsonify({'message': 'Logged in successfully', 'username': username}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'}), 200


@app.route('/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'username': session['username']}), 200
    else:
        return jsonify({'authenticated': False}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    proba = classifier.predict_proba(X)[0]
    prediction = classifier.predict(X)[0]
    confidence = float(max(proba))
    sentiment = 'Positive' if prediction == 1 else 'Negative'

    # Word importance (handle tree-based models and SVMs gracefully)
    word_importance = []
    try:
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
            for word in set(cleaned.split()):
                if word in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[word]
                    word_importance.append({'word': word, 'importance': float(coef[idx])})
        elif hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            for word in set(cleaned.split()):
                if word in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[word]
                    word_importance.append({'word': word, 'importance': float(importances[idx])})
        else:
            # Fallback for models without feature importances
            for word in set(cleaned.split()):
                if word in vectorizer.vocabulary_:
                    word_importance.append({'word': word, 'importance': 0.0})
    except Exception as e:
        print(f"[WARNING] Could not calculate word importance: {e}")

    # Save to history
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, text, sentiment, confidence) VALUES (?, ?, ?, ?)",
              (session['user_id'], text, sentiment, confidence))
    conn.commit()
    conn.close()

    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'proba_pos': float(proba[1]) if len(proba) > 1 else confidence,
        'proba_neg': float(proba[0]),
        'importance': word_importance
    })


@app.route('/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT text, sentiment, confidence, timestamp FROM history "
        "WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
        (session['user_id'],)
    )
    rows = c.fetchall()
    conn.close()

    return jsonify([
        {'text': r[0], 'sentiment': r[1], 'confidence': r[2], 'timestamp': r[3]}
        for r in rows
    ])


@app.route('/clear-history', methods=['POST'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user_id = ?", (session['user_id'],))
    conn.commit()
    conn.close()
    return jsonify({'message': 'History cleared'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)