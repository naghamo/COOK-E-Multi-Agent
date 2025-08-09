"""Agent 8: Order Execution
--------------------------------------
This agent generates supermarket-style PDF receipts for the user's order,
 to simulate a real-world shopping experience and payment presses."""

from fpdf import FPDF

import os
import random
from datetime import datetime


def _clean_code(val):
    """
    Clean product code for display.
    Removes trailing '.0' if present (common for codes stored as floats).

    Args:
        val (str|int|float|None): The product code.

    Returns:
        str: Cleaned code string or empty string if None.
    """
    if val is None:
        return ""
    s = str(val)
    return s[:-2] if s.endswith(".0") else s


def _load_dejavu_fonts(pdf: FPDF, font_dir="static/fonts"):
    """
    Registers DejaVu Sans regular + bold in the PDF object.
    Falls back to regular font for bold if bold file missing.

    Args:
        pdf (FPDF): PDF object.
        font_dir (str): Directory containing font files.

    Raises:
        FileNotFoundError: If regular font file is missing.
    """
    reg_path = os.path.join(font_dir, "DejaVuSans.ttf")
    bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")

    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Missing font file: {reg_path}")
    pdf.add_font("DejaVu", "", reg_path)  # No 'uni' param in recent fpdf2
    if os.path.exists(bold_path):
        pdf.add_font("DejaVu", "B", bold_path)
    else:
        # If bold not found, map bold to regular
        pdf.add_font("DejaVu", "B", reg_path)


def make_supermarket_pdf(store_name, store_info, recipe_title,
                         filename, delivery_mode, user_name, order_number):
    """
    Generates a supermarket-style receipt PDF for a specific store.

    Args:
        store_name (str): Name of the supermarket.
        store_info (dict): Dictionary containing 'items', 'delivery_fee', 'grand_total', etc.
        recipe_title (str): Name of the recipe for which items were purchased.

        filename (str): Path to save the PDF.
        delivery_mode (str): 'Delivery' or 'Pickup'.
        user_name (str): Name of the customer.
        order_number (int): Unique order number.

    Returns:
        str: Path of the saved PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Use Unicode font for ₪ and special characters
    _load_dejavu_fonts(pdf, font_dir="static/fonts")
    pdf.set_font("DejaVu", "", 12)

    # Header section
    pdf.cell(0, 10, f"Order Number: {order_number}", ln=True)
    pdf.cell(0, 10, f"Customer: {user_name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.set_font("DejaVu", "B", 20)
    pdf.cell(0, 15, store_name.upper(), align='C', ln=True)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f"Recipe: {recipe_title}", ln=True)
    pdf.cell(0, 8, f"Delivery Mode: {delivery_mode}", ln=True)
    pdf.ln(4)

    # Table header
    col_widths = [40, 60, 20, 25, 25]
    pdf.set_font("DejaVu", "B", 12)
    headers = ["Code", "Product", "Qty", "Total"]
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, 1)
    pdf.ln()
    pdf.set_font("DejaVu", "", 10)

    # Table rows
    subtotal = 0
    for item in store_info["items"]:
        code = _clean_code(item["code"])
        total_price = item["total_price"]
        subtotal += float(total_price)
        row = [
            code,
            item["product"],
            str(item["qty"]),

            f"{total_price:.2f}₪"
        ]
        for text, w in zip(row, col_widths):
            pdf.cell(w, 8, text, 1)
        pdf.ln()

    # Totals section
    pdf.ln(3)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, f"Subtotal: {subtotal:.2f}₪", ln=True)
    pdf.cell(0, 8, f"Delivery Fee: {store_info['delivery_fee']:.2f}₪ ({delivery_mode})", ln=True)
    pdf.cell(0, 10, f"Grand Total: {store_info['grand_total']:.2f}₪", ln=True)



    pdf.output(filename)
    return filename


def finalize_order_generate_pdfs(stores_dict, recipe_title,
                                 delivery_choices, user_name, pdf_dir='static/receipts'):
    """
    Creates PDF receipts for each supermarket in an order.

    Args:
        stores_dict (dict): Dictionary mapping store names to their purchase info.
       recipe_title (str): Title of the recipe for which items were purchased.
        delivery_choices (dict): Mapping store -> bool (True if delivery, False if pickup).
        user_name (str): Name of the customer.
        pdf_dir (str): Directory to store the generated PDFs.

    Returns:
        dict: Mapping store name -> URL path to the generated PDF.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_links = {}
    order_number = random.randint(100000, 999999)  # Same order number for all stores in this order

    for store, info in stores_dict.items():
        delivery_mode = 'Delivery' if delivery_choices.get(store, True) else 'Pickup'
        filename = f"{pdf_dir}/receipt_{store}_{order_number}.pdf"
        make_supermarket_pdf(
            store_name=store,
            store_info=info,
            recipe_title=recipe_title,

            filename=filename,
            delivery_mode=delivery_mode,
            user_name=user_name,
            order_number=order_number
        )
        pdf_links[store] = '/' + filename
    return pdf_links


if __name__ == "__main__":
    # Example input for testing
    stores_dict = {
        "osher_ad": {
            "delivery_fee": 16.0,
            "grand_total": 70.4,
            "notes": "Selected for majority of items due to cost-effectiveness and promotions.",
            "items": [
                {"code": "7290017065786.0", "product": "Grated Cheddar Cheese 200 g", "qty": 1, "unit_price": 20.9,
                 "total_price": 20.9, "promo": "Buy 1 Get 1"},
                {"code": "6936613888667.0", "product": "4 heads of garlic", "qty": 1, "unit_price": 4.9,
                 "total_price": 4.9, "promo": None}
            ]
        },
        "tiv_taam": {
            "delivery_fee": 18.0,
            "grand_total": 37.9,
            "notes": "Used for tomato sauce as it was unavailable in other stores.",
            "items": [
                {"code": "4750022001993", "product": "Tomato sauce with garlic 510 g SPILVA", "qty": 1,
                 "unit_price": 19.9, "total_price": 19.9, "promo": None}
            ]
        }
    }

    recipe_title = "Vegan Shakshuka"
    recipe_directions = [
        "Heat olive oil in a pan.",
        "Add onions and cook until soft.",
        "Add peppers, tomatoes, and spices.",
        "Simmer until thickened and serve."
    ]
    delivery_choices = {"osher_ad": True, "tiv_taam": False}
    user_name = "Nagham Omar"

    # Test PDF generation
    pdf_links = finalize_order_generate_pdfs(
        stores_dict=stores_dict,
        recipe_title=recipe_title,
        delivery_choices=delivery_choices,
        user_name=user_name
    )
    print(pdf_links)
