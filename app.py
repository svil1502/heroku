from flask import Flask, render_template, request


MAX_FILE_SIZE = 1024 * 1024 + 1

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        if bool(file.filename):
            file_bytes = file.read(MAX_FILE_SIZE)
            args["file_size_error"] = len(file_bytes) == MAX_FILE_SIZE
        args["method"] = "POST"
    return render_template("index.html", args=args)

if __name__ == "__main__":
    app.run(debug=True)