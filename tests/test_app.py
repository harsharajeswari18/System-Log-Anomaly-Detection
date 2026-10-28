import io
from app import app
from ml.pipeline import demo_dataframe,predict_dataframe
def test_health():
 r=app.test_client().get("/health"); assert r.status_code==200 and r.json["status"]=="ok"
def test_pipeline():
 out,s=predict_dataframe(demo_dataframe()); assert len(out)==88; assert "prediction" in out and "anomaly_score" in out; assert s["rows"]==88
def test_home():
 assert app.test_client().get("/").status_code==200
def test_upload():
 c=app.test_client(); payload=b"event_count,duration\n1,0.2\n1.1,0.3\n9,8\n"
 r=c.post("/predict",data={"file":(io.BytesIO(payload),"sample.csv")},content_type="multipart/form-data",follow_redirects=True)
 assert r.status_code==200 and b"Analysis summary" in r.data
