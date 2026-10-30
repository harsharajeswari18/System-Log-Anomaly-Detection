
import io
from app import app
from ml.pipeline import demo_dataframe, predict_dataframe


def test_health():
    r = app.test_client().get("/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"


def test_pipeline():
    out, summary = predict_dataframe(demo_dataframe())
    assert len(out) == 88
    assert "prediction" in out
    assert "anomaly_score" in out
    assert summary["rows"] == 88


def test_home():
    assert app.test_client().get("/").status_code == 200


def test_upload():
    c = app.test_client()
    payload = (
        b"event_count,sequence_length,duration,error_rate\n"
        b"1,2,0.2,0.1\n"
        b"2,3,0.3,0.2\n"
        b"9,10,8.0,0.9\n"
    )

    r = c.post(
        "/predict",
        data={"file": (io.BytesIO(payload), "sample.csv")},
        content_type="multipart/form-data",
        follow_redirects=True
    )

    assert r.status_code == 200
    assert b"prediction" in r.data.lower()
