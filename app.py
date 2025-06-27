import os

import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'

VALID_USERNAME = 'TECHNION'
VALID_PASSWORD = '0000'
INVENTORY_CSV = "data/home_inventory.csv"

def load_inventory():
    if os.path.exists(INVENTORY_CSV):
        return pd.read_csv(INVENTORY_CSV).to_dict(orient='records')
    # Default inventory if file not found
    return [
        {"name": "Tomato", "quantity": 4, "unit": "pieces", "expiry": "2024-07-10"},
        {"name": "Garlic", "quantity": 2, "unit": "cloves", "expiry": "2024-07-03"}
    ]

def save_inventory(rows):
    df = pd.DataFrame(rows)
    df.to_csv(INVENTORY_CSV, index=False)
@app.route('/inventory', methods=['GET', 'POST'])
def inventory():
    if request.method == 'POST':
        names = request.form.getlist('name')
        quantities = request.form.getlist('quantity')
        units = request.form.getlist('unit')
        expiries = request.form.getlist('expiry')
        rows = []
        for n, q, u, e in zip(names, quantities, units, expiries):
            if n.strip():  # Only add if name is not empty
                rows.append({
                    "name": n.strip(),
                    "quantity": q.strip() if q.strip() else 0,
                    "unit": u.strip() if u.strip() else "",
                    "expiry": e.strip() if e.strip() else ""
                })
        save_inventory(rows)
        return redirect(url_for('main'))
    inventory = load_inventory()
    return render_template('main.html', inventory=inventory, user=session['user'])


@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['user'] = username
            return redirect('/main')
        else:
            error = "Invalid username or password."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/main')
def main():
    if 'user' not in session:
        return redirect('/login')
    inventory = load_inventory()

    return render_template('main.html', user=session['user'], inventory=inventory)

if __name__ == '__main__':
    app.run(debug=True)
