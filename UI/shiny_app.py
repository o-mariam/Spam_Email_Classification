from shiny import App, ui, render
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client import neural_client, e5_client, emotion_client


MODEL_API_MAP = {
    "1": neural_client,
    "2": e5_client,
    "3": emotion_client,
}
app_ui = ui.page_fluid(
    ui.h2("Check if an email is spam or not", style="color:black;"),
    ui.input_text_area("textarea", "Email input"),
    ui.input_action_button("Detector", "Detect"),
    ui.input_radio_buttons(
        "model",
        "Models",
        {"1": "Neural", "2": "Setfit E5", "3": "Setfit Emotion"},
    ),
    ui.output_text_verbatim("output"),
    ui.output_text("status_message")

)


def server(input, output, session):

    @output()
    @render.text
    def status_message():

        model_id = input.model()
        client = MODEL_API_MAP.get(model_id)
        if client.is_alive():
            return f"✅ {model_id} server is online."
        else:
            return f"⚠️ {model_id} server is NOT running."

    @output()
    @render.text
    def output():
        email_text=input.textarea()
        model_id=input.model()
        client = MODEL_API_MAP.get(model_id)
        result = client.detect_one(email_text)
        return f"Email: {result}"



app = App(app_ui, server)
