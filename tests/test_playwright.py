"""
Playwright E2E tests for System Log Anomaly Detection.
Run: pytest tests/test_playwright.py -v
Run with browser visible: pytest tests/test_playwright.py -v --headed
"""
import threading, time, sys, os
import pytest
from playwright.sync_api import Page, expect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app as flask_app

PORT = 5099
BASE_URL = f"http://127.0.0.1:{PORT}"


def _run_server():
    flask_app.run(host="127.0.0.1", port=PORT, use_reloader=False, debug=False)


@pytest.fixture(scope="session", autouse=True)
def live_server():
    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    time.sleep(2)
    yield


# TC-01: Home page 
def test_home_page_loads(page: Page):
    page.goto(BASE_URL)
    expect(page).to_have_title("System Log Anomaly Detection")

def test_home_has_upload_form(page: Page):
    page.goto(BASE_URL)
    expect(page.locator("input[type='file']")).to_be_visible()

def test_home_has_demo_button(page: Page):
    page.goto(BASE_URL)
    expect(page.get_by_role("button", name="Run demo dataset")).to_be_visible()

def test_no_results_on_fresh_load(page: Page):
    page.goto(BASE_URL)
    expect(page.locator("text=Analysis summary")).not_to_be_visible()


# TC-02: Health endpoint 
def test_health_endpoint(page: Page):
    res = page.request.get(f"{BASE_URL}/health")
    assert res.status == 200
    assert res.json()["status"] == "ok"


# TC-03: Demo dataset run
def test_demo_shows_results(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("text=Analysis summary")).to_be_visible(timeout=90_000)

def test_demo_shows_svm(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("Linear SVM", timeout=90_000)

def test_demo_shows_random_forest(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("Random Forest", timeout=90_000)

def test_demo_shows_knn(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("KNN", timeout=90_000)

def test_demo_shows_lstm(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("LSTM", timeout=90_000)

def test_demo_shows_dnn(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("DNN", timeout=90_000)

def test_demo_shows_vqc(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("VQC", timeout=90_000)

def test_demo_shows_isolation_forest(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.locator("table").first).to_contain_text("Isolation Forest", timeout=90_000)

def test_demo_download_button_visible(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.get_by_role("button", name="Download predictions CSV")).to_be_visible(timeout=90_000)


# TC-04: CSV Upload
def test_valid_csv_upload(page: Page):
    page.goto(BASE_URL)
    csv = b"event_count,sequence_length,duration,error_rate\n1,2,0.2,0.1\n9,10,8.0,0.9\n2,3,0.3,0.2\n0.5,1,0.1,0.05\n"
    page.locator("input[type='file']").set_input_files({"name":"test.csv","mimeType":"text/csv","buffer":csv})
    page.get_by_role("button", name="Analyze CSV").click()
    expect(page.locator("text=Analysis summary")).to_be_visible(timeout=90_000)

def test_missing_column_shows_error(page: Page):
    page.goto(BASE_URL)
    bad = b"event_count,sequence_length,duration\n1,2,0.3\n2,3,0.4\n"
    page.locator("input[type='file']").set_input_files({"name":"bad.csv","mimeType":"text/csv","buffer":bad})
    page.get_by_role("button", name="Analyze CSV").click()
    expect(page.locator(".message")).to_be_visible(timeout=10_000)

def test_non_csv_shows_error(page: Page):
    page.goto(BASE_URL)
    page.locator("input[type='file']").set_input_files({"name":"file.txt","mimeType":"text/plain","buffer":b"hello"})
    page.get_by_role("button", name="Analyze CSV").click()
    expect(page.locator(".message")).to_be_visible(timeout=10_000)

def test_empty_file_shows_error(page: Page):
    page.goto(BASE_URL)
    page.locator("input[type='file']").set_input_files({"name":"empty.csv","mimeType":"text/csv","buffer":b""})
    page.get_by_role("button", name="Analyze CSV").click()
    expect(page.locator(".message")).to_be_visible(timeout=10_000)


# TC-05: Download
def test_download_csv(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="Run demo dataset").click()
    expect(page.get_by_role("button", name="Download predictions CSV")).to_be_visible(timeout=90_000)
    with page.expect_download() as dl:
        page.get_by_role("button", name="Download predictions CSV").click()
    assert dl.value.suggested_filename == "anomaly_predictions.csv"