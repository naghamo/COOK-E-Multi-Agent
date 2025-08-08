import json
import os

import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for
from pipeline import run_pipeline_to_inventory_confirmation, run_pipeline_to_order_execution
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

@app.template_filter('from_json')
def from_json_filter(s):
    if not s:
        return []
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return [s]
    return s
@app.route('/submit_request', methods=['POST'])
def submit_request():
    if 'user' not in session:
        return redirect(url_for('login'))
    inventory = load_inventory()
    units_list = load_units()
    user_text = request.form.get('user_input')
    # Call the pipeline
    result = run_pipeline_to_inventory_confirmation(user_text, tokens_filename="tokens/total_tokens.txt")
    old_requests = load_old_requests()
    session['last_user_text'] = user_text
    session['last_recipe'] = result.get('recipe', {})

    if 'error' in result:
        # Show error to user in main.html
        return render_template(
            'main.html',
            user=session['user'],
            inventory=inventory,
            old_requests=old_requests,
            error=result['error'],
            units=units_list,
            user_input=user_text,
            request=False
        )

    # ---- ADD TO OLD REQUESTS ----
    request_data = {
        "user": session['user'],
        "user_text": user_text,
        "recipe_title": result['recipe'].get('title', ''),
        "directions": json.dumps(result['recipe'].get('directions', []), ensure_ascii=False),
        "recipe_servings": result['context'].get('people', ''),
        "recipe_ingredients": json.dumps(result['ingredients'], ensure_ascii=False),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_old_request(request_data)
    confirmation_list = result['confirmation_json']
    for ing in confirmation_list:
        u = ing.get("to_buy_unit")
        if u and u not in units_list:
            units_list.append(u)
    save_units(units_list)
    # --- Show confirmation to user ---

    return render_template(
        'main.html',
        user=session['user'],
        inventory=inventory,
        old_requests=load_old_requests(),
        confirmation_list=confirmation_list,
        units=units_list,
        user_input=user_text,
        request=True,
    conf_type='conf'
    )


@app.route('/confirm_ingredient', methods=['POST'])
def confirm_ingredient():
    if 'user' not in session:
        return redirect(url_for('login'))

    inventory = load_inventory()
    units_list = load_units()
    old_requests = load_old_requests()
    user_text = session.get('last_user_text', '')  # How you save user input
    recipe = session.get('last_recipe', {})        # How you save recipe object

    # 1. Get confirmed ingredient info from form
    confirmed_ingredients = []
    i = 0
    while True:
        name = request.form.get(f"name_{i}")
        if not name:
            break
        to_buy_min = float(request.form.get(f"to_buy_min_{i}", 0))
        to_buy_unit = request.form.get(f"to_buy_unit_{i}")
        confirmed_ingredients.append({
            "name": name,
            "to_buy_min": to_buy_min,
            "to_buy_unit": to_buy_unit
        })
        i += 1
        print(confirmed_ingredients)

    # 2. Check if everything is at home
    # nothing_to_buy = all(item["to_buy_min"] == 0 for item in confirmed_ingredients)
    # if nothing_to_buy:
    #     recipe_title = recipe.get('title', '')
    #     recipe_servings = recipe.get('servings', '')
    #     recipe_ingredients = recipe.get('ingredients', [])  # as list of strings
    #     recipe_directions = recipe.get('directions', [])    # as list of strings
    #
    #     # Save in old requests db
    #     request_data = {
    #         "user": session['user'],
    #         "user_text": user_text,
    #         "recipe_title": recipe_title,
    #         "recipe_servings": recipe_servings,
    #         "recipe_ingredients": json.dumps(recipe_ingredients, ensure_ascii=False),
    #         "directions": json.dumps(recipe_directions, ensure_ascii=False),
    #         "purchase_result": "Nothing to buy - all ingredients at home",
    #         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     }
    #     save_old_request(request_data)
    #
    #     return render_template(
    #         'main.html',
    #         user=session['user'],
    #         inventory=inventory,
    #         old_requests=load_old_requests(),
    #         units=units_list,
    #         user_input=user_text,
    #         recipe_title=recipe_title,
    #         recipe_servings=recipe_servings,
    #         recipe_ingredients=recipe_ingredients,
    #         recipe_directions=recipe_directions,
    #         request=False,
    #         message="All ingredients are available at home! No need to order anything 😊"
    #     )

    # 3. Otherwise, proceed with purchase pipeline
    result = run_pipeline_to_order_execution(confirmed_ingredients,
                                             tokens_filename="tokens/total_tokens.txt")

    if 'error' in result or result.get("not_feasible"):
        # Handle case where no feasible order/cart could be made
        return render_template(
            'main.html',
            user=session['user'],
            inventory=inventory,
            old_requests=old_requests,
            error=result.get('error', "Sorry, the requested order is not feasible."),
            units=units_list,
            user_input=user_text,
            request=True,
        )

    # If "cart" is empty, show a special message or the recipe summary
    if not result.get("cart"):
        return render_template(
            'main.html',
            user=session['user'],
            inventory=inventory,
            old_requests=old_requests,
            units=units_list,
            user_input=user_text,
            recipe_title=result.get('recipe_title', ''),
            recipe_directions=result.get('directions', []),
            recipe_ingredients=result.get('ingredients', []),
            nothing_to_buy=True,  # special flag for template
            request=True,
        )

    # Normal payment confirmation view:
    return render_template(
        'main.html',
        user=session['user'],
        inventory=inventory,
        old_requests=old_requests,
        stores=result['stores'],  # dict of stores
        total_payment=result['total_payment'],  # grand total of all stores
        payment_conf=True,
        units=units_list,
        user_input=user_text,
        request=True,
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
