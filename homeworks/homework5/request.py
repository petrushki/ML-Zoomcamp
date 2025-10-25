import requests

url = "http://localhost:9696/predict"

customer = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

pred = requests.post(url, json=customer)


print(pred.status_code)
print(pred.json())
