from shiny import App, ui, render
import
MODEL_API_MAP = {
    "1": "http://localhost:5000/detect",
    "2": "http://localhost:5001/detect",
    "3": "http://localhost:5002/detect",
}
app_ui = ui.page_fluid(
    ui.h2("Check if an email is spam or not", style="color:black;"),
    ui.input_text_area("textarea", "Email input", "Email text"),
    ui.input_action_button("Detector", "Detect"),
    ui.input_radio_buttons(
        "model",
        "Models",
        {"1": "Neural", "2": "Setfit E5", "3": "Setfit Emotion"},
    ),
    ui.output_text_verbatim("output")
)


def server(input, output, session):
    @output()
    @render.text
    def output():
        # if input.Detector()==0:
        email_text=input.textarea()
        model_id=input.model()



        return f"Email: {input.textarea()}"


app = App(app_ui, server)
