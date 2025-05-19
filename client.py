import requests
import json


class SpamDetectorClient:
    def _init_(self, server_url="http://localhost:5000"):
        self.server_url = server_url

    def info(self):
        info_url = ''
        api_url = f"{self.server_url}/{info_url}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    def detect_one(self, email_text: str):
        detect_one_url = ''
        api_url = f"{self.server_url}/{detect_one_url}"
        payload = {}
        return False

    def detect_many(self, email_texts: list[str]):
        detect_many_url = ''
        api_url = f"{self.server_url}/{detect_many_url}"
        payload = {}
        return [False] * len(email_texts)


if _name_ == '_main_':
    client = SpamDetectorClient()
    print(client.info())
    print(client.detect_one("This is a test email."))
    print(client.detect_many(["This is a test email.", "Another test email."]))