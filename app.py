import io, os
import pandas as pd
from flask import Flask, request, render_template_string, send_file, flash, redirect, url_for
from ml.pipeline import predict_dataframe, demo_dataframe
app=Flask(__name__); app.secret_key=os.getenv("SECRET_KEY","local-demo-only"); app.config["MAX_CONTENT_LENGTH"]=10*1024*1024
HTML="""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>System Log Anomaly Detection</title><style>body{font-family:Arial;background:#f3f5f9;color:#172033;margin:0}header{background:#172554;color:white;padding:22px}main{max-width:1000px;margin:24px auto;padding:0 16px}.card{background:white;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 2px 8px #0001}button{background:#1d4ed8;color:white;border:0;padding:10px 16px;border-radius:7px;cursor:pointer}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px;font-size:13px;text-align:left}th{background:#eaf0ff}.bad{color:#b91c1c}</style></head><body><header><h1>System Log Anomaly Detection</h1><p>Upload numeric log-feature CSVs to flag unusual rows.</p></header><main>{% for m in get_flashed_messages() %}<p>{{m}}</p>{% endfor %}<div class=card><h2>Demo</h2><form action=/demo method=post><button>Run demo dataset</button></form></div><div class=card><h2>Upload CSV</h2><p>Numeric features required. Optional label column: label (Normal/Anomaly or 0/1).</p><form action=/predict method=post enctype=multipart/form-data><input type=file name=file accept=.csv required><button>Analyze CSV</button></form></div>{% if s %}<div class=card><h2>Analysis summary</h2><p>Rows: <b>{{s.rows}}</b> | Anomalies: <b class=bad>{{s.anomalies}}</b></p><p>{{s.method}}</p><form action=/download method=post><input type=hidden name=csv_data value='{{csv}}'><button>Download predictions CSV</button></form></div><div class=card><h2>Preview (first 100 rows)</h2>{{table|safe}}</div>{% endif %}<div class=card><b>Note:</b> This sprint MVP uses numeric features and Isolation Forest; it does not parse raw HDFS logs automatically.</div></main></body></html>"""
def render(result=None):
 return render_template_string(HTML,s=result["summary"] if result else None,table=result["table"] if result else None,csv=result["csv"] if result else "")
def analyze(df):
 out,s=predict_dataframe(df); return {"summary":s,"table":out.head(100).to_html(index=False,escape=True),"csv":out.to_csv(index=False)}
@app.get("/")
def home(): return render()
@app.get("/health")
def health(): return {"status":"ok"},200
@app.post("/demo")
def demo():
 try:return render(analyze(demo_dataframe()))
 except Exception as e: flash(str(e)); return redirect(url_for("home"))
@app.post("/predict")
def predict():
 f=request.files.get("file")
 if not f or not f.filename.lower().endswith(".csv"): flash("Please upload a CSV file."); return redirect(url_for("home"))
 try:
  df=pd.read_csv(f)
  if df.empty: raise ValueError("CSV contains no data rows.")
  if len(df)>50000: raise ValueError("Maximum 50,000 rows per upload.")
  return render(analyze(df))
 except Exception as e: flash("Could not analyze CSV: "+str(e)); return redirect(url_for("home"))
@app.post("/download")
def download():
 data=request.form.get("csv_data","")
 if not data: flash("No results to download."); return redirect(url_for("home"))
 return send_file(io.BytesIO(data.encode()),mimetype="text/csv",as_attachment=True,download_name="anomaly_predictions.csv")
if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")),debug=False)
