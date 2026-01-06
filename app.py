from ui import create_ui

if __name__ == "__main__":
    print("Starting Video Query System...")
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)
