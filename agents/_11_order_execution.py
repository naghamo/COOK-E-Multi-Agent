from fpdf import FPDF
import os

def make_supermarket_pdf(store_name, store_info, recipe_title, recipe_directions, filename, delivery_mode, user_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.add_paragraph(f"Order for: {user_name}")
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, store_name.upper(), align='C', ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Recipe: {recipe_title}", ln=True)
    pdf.cell(0, 8, f"Delivery Mode: {delivery_mode}", ln=True)
    pdf.ln(4)


    # Table header
    pdf.set_font("Arial", "B", 13)
    pdf.cell(40, 8, "Code", 1)
    pdf.cell(60, 8, "Product", 1)
    pdf.cell(20, 8, "Qty", 1)
    pdf.cell(25, 8, "Unit Price", 1)
    pdf.cell(25, 8, "Total", 1)
    pdf.ln()
    pdf.set_font("Arial", "", 12)
    subtotal = 0

    for item in store_info["items"]:
        code = str(item["code"])
        if code.endswith(".0"):
            code = code[:-2]
        total_price = item["total_price"]
        subtotal += float(total_price)
        pdf.cell(40, 8, code, 1)
        pdf.cell(60, 8, item["product"], 1)
        pdf.cell(20, 8, str(item["qty"]), 1)
        pdf.cell(25, 8, f"{item['unit_price']}₪", 1)
        pdf.cell(25, 8, f"{total_price}₪", 1)
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Subtotal: {subtotal:.2f}₪", ln=True)
    pdf.cell(0, 8, f"Delivery Fee: {store_info['delivery_fee']:.2f}₪ ({delivery_mode})", ln=True)
    pdf.cell(0, 10, f"Grand Total: {store_info['grand_total']:.2f}₪", ln=True)

    # Optional: more details or store notes
    if store_info.get("notes"):
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 8, f"Note: {store_info['notes']}")
    pdf.ln(5)

    # Recipe directions at the end
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recipe Directions:", ln=True)
    pdf.set_font("Arial", "", 11)
    for idx, step in enumerate(recipe_directions):
        pdf.multi_cell(0, 8, f"{idx+1}. {step}")
        pdf.ln(1)

    pdf.output(filename)
    return filename
def finalize_order_generate_pdfs(stores_dict, recipe_title, recipe_directions, delivery_choices, user_name, pdf_dir='static/receipts'):
    # Ensure PDF directory exists
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_links = {}
    for store, info in stores_dict.items():
        delivery_mode = 'Delivery' if delivery_choices.get(store, True) else 'Pickup'
        # Optional: get LLM summary
        # summary = make_llm_store_summary(store, info['items'], delivery_mode, chat_llm)
        filename = f"{pdf_dir}/receipt_{store}.pdf"
        make_supermarket_pdf(
            store_name=store,
            store_info=info,
            recipe_title=recipe_title,
            recipe_directions=recipe_directions,
            filename=filename,
            delivery_mode=delivery_mode,
            user_name=user_name

        )
        # Link for web
        pdf_links[store] = '/' + filename
    return pdf_links