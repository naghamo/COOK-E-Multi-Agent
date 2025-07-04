import os

import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for
from pipeline import run_cooke_pipeline
from datetime import datetime
app = Flask(__name__)
app.secret_key = 'your_secret_key'

VALID_USERNAME = 'TECHNION'
VALID_PASSWORD = '0000'
INVENTORY_CSV = "data/home_inventory.csv"
OLD_REQUESTS_CSV = "data/old_requests.csv"

UNITS_FILE = "data/units.txt"

def load_units():
    if not os.path.exists(UNITS_FILE):
        return ['units', 'grams', 'kg', 'ml', 'l']
    with open(UNITS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

def save_units(units_list):
    with open(UNITS_FILE, "w") as f:
        f.write("\n".join(sorted(set(units_list))))

def load_old_requests():
    if os.path.exists(OLD_REQUESTS_CSV):
        return pd.read_csv(OLD_REQUESTS_CSV).to_dict(orient='records')
    return []

def save_old_request(request_data):
    all_requests = load_old_requests()
    all_requests.insert(0, request_data)  # newest first
    df = pd.DataFrame(all_requests)
    df.to_csv(OLD_REQUESTS_CSV, index=False)
def load_inventory():
    if os.path.exists(INVENTORY_CSV):
        #check if the file is not empty
        if os.path.getsize(INVENTORY_CSV) > 0:
            # Load the inventory from CSV

            return pd.read_csv(INVENTORY_CSV).to_dict(orient='records')
        else:
            # If file is empty, return empty inventory with columns
            print("Inventory file is empty, returning default inventory.")
            return [{"name": "", "quantity": 0, "unit": "units", "expiry": ""}]

    # Default inventory if file not found
    return [
        {"name": "Tomato", "quantity": 4, "unit": "pieces", "expiry": "2024-07-10"},
        {"name": "Garlic", "quantity": 2, "unit": "cloves", "expiry": "2024-07-03"}
    ]

def save_inventory(rows):
    df = pd.DataFrame(rows)
    df.to_csv(INVENTORY_CSV, index=False)




@app.route('/submit_request', methods=['POST'])
def submit_request():
    if 'user' not in session:
        return redirect(url_for('login'))
    inventory = load_inventory()
    units_list = load_units()
    user_text = request.form.get('user_input')
    # Call the pipeline
    # first_conf_req,conf_type = run_cooke_pipeline(user_text, inventory)
    old_requests = load_old_requests()
    #add the request to old requests
    # request_data = {
    #     "user": session['user'],
    #     "text": user_text,
    #     "result": result,
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # }
    # save_old_request(request_data)

    first_conf_req = [{
        'name': 'tomato',
        'requested_quantity': 4,
        'requested_unit': 'pcs',
        'to_buy_min': 3,
        'to_buy_unit': 'pcs',
        'editing': False
    }]
    conf_type = "conf"

    # Render main.html with all context
    return render_template(
        'main.html',
        user=session['user'],
        inventory=inventory,
        old_requests=old_requests,
        first_conf_req=first_conf_req,
        conf_type=conf_type,
        units=units_list,
        user_input=user_text,
        request=True
    )

@app.route('/confirm_ingredient', methods=['POST'])
def confirm_ingredient():
    if 'user' not in session:
        return redirect(url_for('login'))
    inventory = load_inventory()
    units_list = load_units()
    old_requests = load_old_requests()

    # 1. Get confirmed ingredient info from form
    # Let's assume first_conf_req is stored in session (or get from request.form as above)
    confirmed_ingredients = []
    i = 0
    while True:
        name = request.form.get(f"name_{i}")
        if not name:
            break
        to_buy_min = request.form.get(f"to_buy_min_{i}")
        to_buy_unit = request.form.get(f"to_buy_unit_{i}")
        confirmed_ingredients.append({
            "name": name,
            "to_buy_min": float(to_buy_min),
            "to_buy_unit": to_buy_unit
        })
        i += 1

    # 2. Run next pipeline step (product matcher, market selector, etc)
    # You should implement or call something like:
    #   market_name, delivery, cart, promos, total_price = run_checkout_pipeline(confirmed_ingredients)
    # For this example, let's mock it:

    market_name = "Shufersal"
    delivery = {
        "fee": 15.9,
        "time": "Today 18:00-20:00"
    }
    cart = [
        {
            "product": "Fresh Tomatoes 1kg",
            "code": "12345678",
            "brand": "Shufersal",
            "qty": 1,
            "unit_price": 9.90,
            "total_price": 9.90,
            "promo": "5% off"
        },
        {
            "product": "Olive Oil 750ml",
            "code": "22334455",
            "brand": "Yad Mordechai",
            "qty": 1,
            "unit_price": 22.90,
            "total_price": 22.90,
            "promo": ""
        }
    ]
    total_price = sum([item["total_price"] for item in cart]) + delivery["fee"]

    # 3. Render payment confirmation page
    return render_template(
        'main.html',
        user=session['user'],
        inventory=inventory,
        old_requests=old_requests,
        supermarket=market_name,
        delivery=delivery,
        cart=cart,
        total_price=total_price,
        payment_conf=True,
        units=units_list,
    )

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
    units_list = load_units()
    old_requests = load_old_requests()
    if request.method == 'POST':
        # Handle delete action
        if 'delete_idx' in request.form:
            delete_idx = int(request.form['delete_idx'])
            if 0 <= delete_idx < len(inventory):
                del inventory[delete_idx]
                save_inventory(inventory)
            return redirect(url_for('main'))

        # Update existing inventory
        names = request.form.getlist('name')
        quantities = request.form.getlist('quantity')
        units = request.form.getlist('unit')
        custom_units = request.form.getlist('custom_unit')
        expiries = request.form.getlist('expiry')
        rows = []

        for n, q, u, cu, e in zip(names, quantities, units, custom_units, expiries):
            # If "other" was selected, use the custom unit value
            real_unit = cu.strip() if u == 'other' and cu.strip() else u.strip()
            if real_unit and real_unit not in units_list:
                units_list.append(real_unit)

            real_unit = cu.strip() if u == 'other' and cu.strip() else u.strip()
            if n.strip():
                rows.append({
                    "name": n.strip(),
                    "quantity": q.strip() if q.strip() else 0,
                    "unit": real_unit if real_unit else "units",
                    "expiry": e.strip() if e.strip() else ""
                })
        save_units(units_list)
        # For new item
        new_name = request.form.get('new_name', '').strip()
        new_quantity = request.form.get('new_quantity', '').strip()
        new_unit = request.form.get('new_unit', '').strip()
        new_custom_unit = request.form.get('new_custom_unit', '').strip()
        new_real_unit = new_custom_unit if new_unit == 'other' and new_custom_unit else new_unit
        if new_name:
            rows.append({
                "name": new_name,
                "quantity": new_quantity if new_quantity else 0,
                "unit": new_real_unit if new_real_unit else "units",
                "expiry": request.form.get('new_expiry', '').strip() or ""
            })

        save_inventory(rows)
        return redirect(url_for('main'))

    return render_template('main.html', inventory=inventory, user=session['user'], units=units_list,old_requests=old_requests,request=False)



if __name__ == '__main__':
    app.run(debug=True)
