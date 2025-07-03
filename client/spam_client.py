import requests
import json



class SpamDetectorClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def info(self):
        api_url = f"{self.server_url}/{'model/info'}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    def detect_one(self, email_text: str):
        api_url = f"{self.server_url}/{'model/email'}"
        headers = {"Content-Type": "application/json; charset=utf-8"}
        data={"email_text": email_text}
        response = requests.post(api_url,headers=headers, json=data)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    def detect_many(self, email_texts: list[str]):
        headers = {"Content-Type": "application/json; charset=utf-8"}
        api_url = f"{self.server_url}/{'model/emails'}"
        data = {"email_texts": email_texts}
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    def is_alive(self):
        try:
            response = requests.get(f"{self.server_url}/alive",timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False


if __name__ == '__main__':
    pass
