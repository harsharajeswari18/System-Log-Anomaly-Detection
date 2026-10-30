
import csv
import io
import os

import pandas as pd
from flask import (
    Flask, request, render_template_string, send_file,
    flash, redirect, url_for
)
from werkzeug.exceptions import RequestEntityTooLarge

from ml.pipeline import predict_dataframe, demo_dataframe


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "local-demo-only")

# Upload and dataset limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 50000

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Required columns for the ML pipeline
REQUIRED_COLUMNS = [
    "event_count",
    "sequence_length",
    "duration",
    "error_rate"
]

OPTIONAL_COLUMNS = ["label"]


HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>System Log Anomaly Detection</title>
    <style>
        body {
            font-family: Arial;
            background: #f3f5f9;
            color: #172033;
            margin: 0;
        }
        header {
            background: #172554;
            color: white;
            padding: 22px;
        }
        main {
            max-width: 1000px;
            margin: 24px auto;
            padding: 0 16px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 2px 8px #0001;
        }
        button {
            background: #1d4ed8;
            color: white;
            border: 0;
            padding: 10px 16px;
            border-radius: 7px;
            cursor: pointer;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        td, th {
            border: 1px solid #ddd;
            padding: 7px;
            font-size: 13px;
            text-align: left;
        }
        th {
            background: #eaf0ff;
        }
        .bad {
            color: #b91c1c;
        }
        .message {
            background: #fff1f2;
            color: #991b1b;
            padding: 12px;
            border-radius: 7px;
            margin: 10px 0;
        }
    </style>
</head>

<body>
<header>
    <h1>System Log Anomaly Detection</h1>
    <p>Upload numeric log-feature CSVs to identify unusual patterns.</p>
</header>

<main>

    {% for message in get_flashed_messages() %}
        <div class="message">{{ message }}</div>
    {% endfor %}

    <div class="card">
        <h2>Demo</h2>
        <form action="/demo" method="post">
            <button type="submit">Run demo dataset</button>
        </form>
    </div>

    <div class="card">
        <h2>Upload CSV</h2>

        <p>Maximum file size: 10 MB</p>
        <p>Maximum rows: 50,000</p>

        <p>Required numeric columns:</p>
        <ul>
            {% for col in required_columns %}
                <li>{{ col }}</li>
            {% endfor %}
        </ul>

        <p>Optional column: label</p>

        <form action="/predict" method="post"
              enctype="multipart/form-data">

            <input type="file" name="file"
                   accept=".csv" required>

            <button type="submit">Analyze CSV</button>
        </form>
    </div>

    {% if s %}
    <div class="card">
        <h2>Analysis summary</h2>

        <p>Rows: <b>{{ s.rows }}</b></p>
        <p>
            Anomalies:
            <b class="bad">{{ s.anomalies }}</b>
        </p>
        <p>{{ s.method }}</p>

        <form action="/download" method="post">
            <input type="hidden" name="csv_data" value="{{ csv }}">
            <button type="submit">Download predictions CSV</button>
        </form>
    </div>

    <div class="card">
        <h2>Preview (first 100 rows)</h2>
        {{ table|safe }}
    </div>
    {% endif %}

    <div class="card">
        <b>Note:</b>
        This version uses numeric features and an Isolation Forest
        baseline. Raw HDFS log parsing is not yet implemented.
    </div>

</main>
</body>
</html>
"""


def render(result=None):
    return render_template_string(
        HTML,
        s=result["summary"] if result else None,
        table=result["table"] if result else None,
        csv=result["csv"] if result else "",
        required_columns=REQUIRED_COLUMNS
    )


def analyze(df):
    output, summary = predict_dataframe(df)

    return {
        "summary": summary,
        "table": output.head(100).to_html(
            index=False,
            escape=True
        ),
        "csv": output.to_csv(index=False)
    }


def validate_csv(file):
    """
    Validate uploaded CSV file before sending it to the ML pipeline.
    """

    if not file or not file.filename:
        raise ValueError("Please select a CSV file.")

    if not file.filename.lower().endswith(".csv"):
        raise ValueError("Invalid file type. Please upload a .csv file.")

    raw_data = file.read()

    if not raw_data:
        raise ValueError("The uploaded CSV file is empty.")

    if len(raw_data) > MAX_FILE_SIZE:
        raise ValueError("File size exceeds the 10 MB limit.")

    try:
        text_data = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(
            "Invalid file encoding. Please upload a UTF-8 encoded CSV."
        )

    try:
        reader = csv.reader(io.StringIO(text_data))
        header = next(reader, None)
    except csv.Error:
        raise ValueError("Could not read the CSV header.")

    if not header:
        raise ValueError("CSV header is missing.")

    # Remove spaces and normalize column names
    normalized_header = [
        column.strip().lower() for column in header
    ]

    if any(not column for column in normalized_header):
        raise ValueError("CSV contains an empty column name.")

    if len(normalized_header) != len(set(normalized_header)):
        raise ValueError("CSV contains duplicate column names.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in normalized_header
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    try:
        df = pd.read_csv(io.BytesIO(raw_data))
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ValueError(
            "Invalid CSV format. Please check the rows and delimiters."
        )

    # Normalize DataFrame column names too
    df.columns = [str(column).strip().lower() for column in df.columns]

    if df.empty:
        raise ValueError("CSV contains no data rows.")

    if len(df) > MAX_ROWS:
        raise ValueError(
            "CSV contains more than 50,000 rows. "
            "Please upload a smaller file."
        )

    # Check required numeric columns
    for column in REQUIRED_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")

        invalid_rows = converted[
            converted.isna()
        ].index.tolist()

        if invalid_rows:
            row_numbers = [str(index + 2) for index in invalid_rows[:5]]

            raise ValueError(
                f"Column '{column}' contains missing or non-numeric "
                f"values at CSV row(s): {', '.join(row_numbers)}."
            )

        if not converted.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all():
            raise ValueError(
                f"Column '{column}' contains infinite values."
            )

        df[column] = converted

    return df


@app.get("/")
def home():
    return render()


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.post("/demo")
def demo():
    try:
        return render(analyze(demo_dataframe()))
    except Exception:
        app.logger.exception("Demo analysis failed")
        flash("Demo analysis failed. Please try again.")
        return redirect(url_for("home"))


@app.post("/predict")
def predict():
    try:
        uploaded_file = request.files.get("file")
        dataframe = validate_csv(uploaded_file)

        return render(analyze(dataframe))

    except ValueError as error:
        flash(str(error))
        return redirect(url_for("home"))

    except Exception:
        app.logger.exception("CSV analysis failed")
        flash(
            "Could not analyze the CSV. "
            "Please check the file and try again."
        )
        return redirect(url_for("home"))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    flash("File size exceeds the 10 MB upload limit.")
    return redirect(url_for("home"))


@app.post("/download")
def download():
    data = request.form.get("csv_data", "")

    if not data:
        flash("No results available to download.")
        return redirect(url_for("home"))

    return send_file(
        io.BytesIO(data.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="anomaly_predictions.csv"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
