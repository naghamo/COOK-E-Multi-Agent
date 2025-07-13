# COOK·E – Cognitive Organized Online Kitchen Expert

COOK·E is an **autonomous, multi-agent AI system** that transforms a user's free-text cooking request into an optimized, ready-to-order grocery cart from real supermarkets. The project demonstrates the power of LLM-based reasoning, product/recipe data integration, and step-by-step pipeline orchestration—all wrapped in a user-friendly web interface.

---

## 💡 Features

* **Free-form input:** User writes any cooking request ("Vegan shakshuka for 4, no tofu, under 70 NIS, delivery").
* **Recipe generation:** Retrieves or creates suitable recipes matching constraints.
* **Inventory-aware:** Uses your home inventory to avoid redundant purchases.
* **Product & supermarket matching:** Finds real products in Israeli supermarkets, compares prices, and optimizes for preferences.
* **Order simulation:** Shows you a detailed “receipt” and simulates order execution.
* **Modular agents:** Each task is handled by a dedicated agent for explainability, modularity, and rapid development.
* **Modern web app:** Secure login, easy editing of home inventory, request history, and results.

---

## 🗂️ Project Structure

```
COOK-E_Agent/
│
├── agents/
│   ├── llm_context_parser.py
│   ├── recipe_retriever.py
│   ├── recipe_parser.py
│   ├── feasibility_checker.py
│   ├── inventory_filter.py
│   ├── product_matcher.py
│   ├── market_selector.py
│   ├── cart_delivery_validator.py
│   ├── order_confirmation.py
│   ├── order_execution.py
│   └── __init__.py
│
├── data/
│   ├── home_inventory.csv
│   ├── old_requests.csv
│   ├── israeli_supermarkets.csv
│   ├── supermarkets.csv
│   ├── unique_products_with_latest_prices.csv
│   ├── israeli_food_brands_from_openfoodfacts.csv
│   └── units.txt
│
├── static/
│   └── style.css
│
├── templates/
│   ├── login.html
│   └── main.html
│
├── app.py
├── pipeline.py
├── requirements.txt
├── README.md
└── tokens_count.py
```

---

## ⚙️ How It Works

1. **Login:** Secure login (user: TECHNION, pass: 0000).
2. **Request:** Enter a free-text cooking request (can include preferences, budget, delivery, allergies, brands, etc).
3. **Pipeline:** Each agent handles a step (input parsing → recipe finding → feasibility → scaling → inventory matching → product matching → store selection → cart validation → confirmation → simulated execution).
4. **Inventory:** Edit your home inventory in the web interface.
5. **Results:** See a detailed receipt and recommendations, all tailored to your constraints and inventory.

---

## 🧠 Agents (Core Modules)

* **LLM Context Parser Agent:** Extracts all structured constraints (dish, servings, requests) from raw user text.
* **Recipe Retriever or Generator Agent:** Finds/generates the best-matching recipe.
* **Feasibility Checker Agent:** Validates recipe existence for constraints (before product matching).
* **Recipe Parser Agent:** Converts recipe text to ingredients with correct quantities.
* **User Ingredient Confirmation Agent:** Cross-checks inventory, proposes items to buy, and requests user confirmation.
* **Inventory Filter Agent:** Filters the final buy-list by confirmation and inventory.
* **Product Matcher Agent:** Maps needed ingredients to real supermarket products.
* **Market Selector Agent:** Optimizes store(s) for cost, delivery, promotions, and preferences.
* **Cart and Delivery Feasibility Validator Agent:** Checks cart/delivery against all constraints; suggests fixes if needed.
* **Order Confirmation Agent:** User sees receipt and approves (or edits).
* **Order Execution Agent:** Simulates checkout, updates inventory, and displays the final recipe.

---

## 📊 Data

* **RecipeNLG Dataset:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/recipenlg)
* **Israeli Supermarkets Product Data:** [Kaggle](https://www.kaggle.com/datasets/erlichsefi/israeli-supermarkets-2024)
* **OpenFoodFacts API** for nutrition/sustainability.
* **Order/Payment APIs** (simulated).

---

## 🔧 Setup

1. `pip install -r requirements.txt`
2. Download required datasets (see `/data/` folder).
3. Run the app:
   `python app.py`
4. Go to [http://localhost:5000](http://localhost:5000), login with the default user, and start cooking!

---

## 📝 Notes

* All LLM-powered agents are tracked for token usage in `tokens_count.py`.
* Agents are modular—easy to expand, improve, or swap for better models/APIs.
* All reasoning, substitutions, and choices are explainable at each step.
* All LLM-powered agents use GPT-4o

---

## 📚 Authors

* Nagham Omar
* Vsevolod Rusanov

