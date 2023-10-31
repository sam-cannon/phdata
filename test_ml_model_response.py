import pytest
import httpx

# The URL where your FastAPI app is running
BASE_URL = "http://127.0.0.1:8000"  # Update with your app's URL

# Path to the CSV file you want to use for testing
CSV_FILE_PATH = "/Users/charlenehack/Desktop/Sam/fastapi_docker_ml/future_unseen_examples.csv"

@pytest.fixture
async def client():
    async with httpx.AsyncClient(app=httpx.app, base_url=BASE_URL) as client:
        yield client

@pytest.mark.asyncio
async def test_prediction_endpoint():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        data = {"file": ("test.csv", open(CSV_FILE_PATH, "rb"))}
        response = await client.post("/predict/", files=data)

    assert response.status_code == 200

    response_data = response.json()
    assert "predictions" in response_data

if __name__ == "__main__":
    pytest.main([__file__])
