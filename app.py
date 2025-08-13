"""
This is the main Flask application for the recipe management system.
It handles user authentication, inventory management, recipe requests, and order processing insted of the terminal.
It uses a pipeline of agents to process user requests, confirm inventory, and execute orders.
"""
import warnings
import importlib
warnings.filterwarnings(
    "ignore",
    message="LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph"
)
# Dynamically import LangChainDeprecationWarning without importing the rest of LangChain yet
LangChainDeprecationWarning = getattr(
    importlib.import_module("langchain_core._api"),
    "LangChainDeprecationWarning"
)

# Hide LangChain deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Optionally hide all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import math
import os

import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for
from pipeline import run_pipeline_to_inventory_confirmation, run_pipeline_to_order_confirmation, \
    run_pipeline_to_order_execution
from datetime import datetime
import pprint
from uuid import uuid4
app = Flask(__name__)
app.secret_key = 'your_secret_key'

VALID_USERNAME = 'TECHNION'
VALID_PASSWORD = '0000'
INVENTORY_CSV = "data/home_inventory.csv"
OLD_REQUESTS_CSV = "data/old_requests.csv"

UNITS_FILE = "data/units.txt"
def update_old_request_by_id(request_id, update_fields):
    '''Updates an old request by its ID in the CSV file.'''
    if not os.path.exists(OLD_REQUESTS_CSV):
        return False
    df = pd.read_csv(OLD_REQUESTS_CSV)
    if 'request_id' not in df.columns:
        return False
    mask = df['request_id'] == request_id
    if not mask.any():
        return False
    for k, v in update_fields.items():
        if k not in df.columns:
            # add missing column if needed
            df[k] = ""
        df.loc[mask, k] = v
    df.to_csv(OLD_REQUESTS_CSV, index=False)
    return True

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
            # print("Inventory file is empty, returning default inventory.")
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
    session['context'] = result.get('context', {})
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
    request_id = str(uuid4())
    session['last_request_id'] = request_id

    request_data = {
        "request_id": request_id,
        "user": session['user'],
        "user_text": user_text,
        "recipe_title": result['recipe'].get('title', ''),
        "directions": json.dumps(result['recipe'].get('directions', []), ensure_ascii=False),
        "recipe_servings": result['context'].get('people', ''),
        "recipe_ingredients": json.dumps(result['ingredients'], ensure_ascii=False),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_old_request(request_data)
    session['last_recipe_title'] = result['recipe'].get('title', []),
    session['last_recipe_directions'] = result['recipe'].get('directions', [])
    session['last_recipe_ingredients'] = result['recipe'].get('ingredients', [])
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

    conf_type='conf',
        recipe_title= result['recipe'].get('title', []),
        recipe_ingredients= result['recipe'].get('ingredients', []),
        recipe_directions= result['recipe'].get('directions', []),
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

    # 1. Get confirmed ingredient info from the form
    confirmed_ingredients = []
    i = 0
    while True:
        name = request.form.get(f"name_{i}")
        if not name:
            break
        to_buy_min = float(request.form.get(f"to_buy_min_{i}", 0))
        to_buy_unit = request.form.get(f"to_buy_unit_{i}")
        if to_buy_min>0:
            confirmed_ingredients.append({
                "name": name,
                "to_buy_min": to_buy_min,
                "to_buy_unit": to_buy_unit
            })
        i += 1
    # print(confirmed_ingredients)

    # 2. Check if everything is at home

    if len(confirmed_ingredients) == 0:
        recipe_title = recipe.get('title', [])
        recipe_servings = recipe.get('servings', '')
        recipe_ingredients = recipe.get('ingredients', [])  # as list of strings
        recipe_directions = recipe.get('directions', [])    # as list of strings

        # Save in old requests db
        request_data = {
            "user": session['user'],
            "user_text": user_text,
            "recipe_title": recipe_title,
            "recipe_servings": recipe_servings,
            "recipe_ingredients": json.dumps(recipe_ingredients, ensure_ascii=False),
            "directions": json.dumps(recipe_directions, ensure_ascii=False),
            "purchase_result": "Nothing to buy - all ingredients at home",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_old_request(request_data)

        return render_template(
            'main.html',
            user=session['user'],
            inventory=inventory,
            old_requests=load_old_requests(),
            units=units_list,
            user_input=user_text,
            recipe_title=recipe_title,
            recipe_servings=recipe_servings,
            recipe_ingredients=recipe_ingredients,
            recipe_directions=recipe_directions,
            request=True,
            message="All ingredients are available at home! No need to order anything 😊"
        )

    # 3. Otherwise, proceed with purchase pipeline
    result = run_pipeline_to_order_confirmation(confirmed_ingredients, tokens_filename="tokens/total_tokens.txt")
    session['pending_stores'] = result.get('stores', {})

    # print(result)
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
            request=False,
        )
    # if the total payment exceeds the budget, show a warning message
    if session['context'].get('budget') is not None and result['total_payment']>session['context'].get('budget'):

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
            recipe_title=recipe.get('title', []),
            recipe_ingredients=recipe.get('ingredients', []),
            recipe_directions=recipe.get('directions', []),
            message=f"Total payment {result['total_payment']} NIS exceeds your budget of {session['context'].get('budget')} NIS. Sorry, we couldn’t find an order within your budget, but we’ve selected the closest and cheapest option available for you. you can cancel items or delivery to get cheaper order"
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
        recipe_title=recipe.get('title', []),
        recipe_ingredients=recipe.get('ingredients', []),
        recipe_directions=recipe.get('directions', []),
    )
@app.route('/confirm_order', methods=['POST'])
def confirm_order():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    inventory = load_inventory()
    units_list = load_units()
    old_requests = load_old_requests()

    # Pull what we saved earlier
    recipe_title = session.get('last_recipe_title', [])
    recipe_directions = session.get('last_recipe_directions', [])
    pending_stores = session.get('pending_stores', {})  # dict we rendered in payment step
    print("Pending stores:", pending_stores)


    # 1) Delivery choices (per store)
    delivery_choices = {
        store: (request.form.get(f"delivery_{store}") == "yes")
        for store in pending_stores.keys()
    }

    # 2) Rebuild a filtered stores_dict (include only items the user kept checked)
    stores_dict = {}
    for store, info in pending_stores.items():
        kept_items = []
        for idx, item in enumerate(info.get('items', [])):
            # Checkbox name pattern from your template: name="buy_{store}_{loop.index0}"
            if request.form.get(f"buy_{store}_{idx}") == "on":
                kept_items.append(item)

        if not kept_items:
            # If user unchecked everything in a store, skip that store entirely
            continue

        # Recompute totals WITHOUT assuming previous grand_total is still valid
        subtotal = sum(float(x.get('total_price', 0) or 0) for x in kept_items)
        delivery_fee = float(info.get('delivery_fee', 0) or 0) if delivery_choices.get(store, True) else 0.0
        grand_total = subtotal + delivery_fee

        stores_dict[store] = {
            "items": kept_items,
            "delivery_fee": float(info.get('delivery_fee', 0) or 0),  # keep original fee for PDF display
            "grand_total": round(grand_total, 2),
            "notes": info.get("notes", ""),
        }

    # Safety: if all stores were unchecked, send the user back with a friendly message
    if not stores_dict:
        return render_template(
            'main.html',
            user=user,
            inventory=inventory,
            old_requests=old_requests,
            error="You unchecked all items. Please select at least one product to order.",
            units=units_list,
            request=True
        )

    # 3) Generate PDFs (NOTE: do NOT pass recipe_directions here; the function doesn’t accept it)
    pdf_links = run_pipeline_to_order_execution(
        stores_dict=stores_dict,
        recipe_title=recipe_title,
        delivery_choices=delivery_choices,
        user_name=user,
        pdf_dir='static/receipts'
    )

    # 4) Save record of the purchase
    request_id = session.get('last_request_id')

    update_fields = {
        "recipe_title": recipe_title,
        "directions": json.dumps(recipe_directions, ensure_ascii=False),
        "stores": json.dumps(list(stores_dict.keys()), ensure_ascii=False),
        "pdf_links": json.dumps(pdf_links, ensure_ascii=False),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purchase_result": "Order placed",
    }

    updated = False
    if request_id:
        updated = update_old_request_by_id(request_id, update_fields)

    # Fallback if for some reason the row wasn’t found (e.g., very old CSV without request_id)
    if not updated:
        # include request_id so future updates work
        fallback_data = {"request_id": request_id or str(uuid4()), "user": user, **update_fields}
        save_old_request(fallback_data)

    return render_template(
        'main.html',
        user=user,
        inventory=inventory,
        old_requests=load_old_requests(),
        recipe_title=recipe_title,
        recipe_directions=recipe_directions,
        pdf_links=pdf_links,
        payment_done=True,
        request=True,
        units=units_list,
        user_input=session.get('last_user_text', ''),  # How you save user input
        recipe_ingredients= session.get('last_recipe_ingredients', []),
        message="Your order has been placed and paid successfully! You can download the receipts below.",
    )


@app.template_filter('from_json')
def from_json_filter(s):
    # Treat None / NaN as empty
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []

    # Parse JSON strings
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            # Fallback: return the raw string in a list
            return [s]

    # Pass through lists/dicts as-is; everything else -> wrap to list of str
    if isinstance(s, (list, tuple, dict)):
        return s
    return [str(s)]
@app.template_filter('stripdotzero')
def stripdotzero(s):
    s = str(s)
    return s[:-2] if s.endswith('.0') else s

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

@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'user' not in session:
        return redirect('/login')

    inventory = load_inventory()
    units_list = load_units()
    old_requests = load_old_requests()

    if request.method == 'POST':
        # --- Delete action (button submits with name="delete_idx") ---
        if 'delete_idx' in request.form:
            try:
                delete_idx = int(request.form['delete_idx'])
            except (ValueError, TypeError):
                return redirect(url_for('main'))

            if 0 <= delete_idx < len(inventory):
                del inventory[delete_idx]
                save_inventory(inventory)
            return redirect(url_for('main'))

        # --- Only proceed for an explicit "Save Changes" submit ---
        if request.form.get('action') != 'save':
            return redirect(url_for('main'))

        # --- Read existing row inputs (use bracketed keys to match HTML) ---
        names = request.form.getlist('name[]')
        quantities = request.form.getlist('quantity[]')
        units = request.form.getlist('unit[]')
        custom_units = request.form.getlist('custom_unit[]')
        expiries = request.form.getlist('expiry[]')

        rows = []

        # Strong guard: if no rows AND no new item, don't wipe the DB
        if not names and not request.form.get('new_name', '').strip():
            # You may still want to persist units_list expansions from a different form section;
            # if not, remove the next line.
            save_units(units_list)
            return redirect(url_for('main'))

        # --- Update existing rows ---
        for n, q, u, cu, e in zip(names, quantities, units, custom_units, expiries):
            n = (n or '').strip()
            q = (q or '').strip()
            u = (u or '').strip()
            cu = (cu or '').strip()
            e = (e or '').strip()

            if not n:
                # Skip blank-name rows
                continue

            # Resolve "other" units once
            real_unit = (cu if (u == 'other' and cu) else u) or 'units'

            # Expand units list if encountering a new unit
            if real_unit not in units_list:
                units_list.append(real_unit)

            rows.append({
                "name": n,
                "quantity": q or 0,
                "unit": real_unit,
                "expiry": e or ""
            })

        # --- Handle new item row (optional) ---
        new_name = request.form.get('new_name', '').strip()
        if new_name:
            new_quantity = request.form.get('new_quantity', '').strip()
            new_unit = request.form.get('new_unit', '').strip()
            new_custom_unit = request.form.get('new_custom_unit', '').strip()
            new_expiry = request.form.get('new_expiry', '').strip()

            new_real_unit = (new_custom_unit if (new_unit == 'other' and new_custom_unit) else new_unit) or 'units'
            if new_real_unit not in units_list:
                units_list.append(new_real_unit)

            rows.append({
                "name": new_name,
                "quantity": new_quantity or 0,
                "unit": new_real_unit,
                "expiry": new_expiry or ""
            })

        # If rows ended up empty, keep existing inventory instead of wiping
        if not rows:
            save_units(units_list)
            return redirect(url_for('main'))

        # --- Persist updates ---
        save_units(units_list)
        save_inventory(rows)
        return redirect(url_for('main'))

    # --- GET: render page ---
    return render_template(
        'main.html',
        inventory=inventory,
        user=session['user'],
        units=units_list,
        old_requests=old_requests,
        request=False
    )


if __name__ == '__main__':
    app.run(debug=True)
