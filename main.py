"""Legacy Flask prototype retained for reference.

This file is no longer the main application entry point. 

Run python app.py from the repository root to start the current 
application.
"""

from flask import Flask, jsonify, render_template
from switch_anim import TestAnimLoader

app = Flask(__name__)
loader = TestAnimLoader(
    default_json="./vocabs/art_gallery_0_mixamo.json"
)


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/start")
def api_start():
    return jsonify(loader.get_default_payload())


@app.post("/api/end")
def api_end():
    return jsonify(
        loader.load_payload(
            "Headbutt_mixamo_com_frames.json",
            animation_name="headbutt",
            camera_state="end",
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
