import requests
import json


class SpamDetectorClient:
    def __init__(self, server_url="http://127.0.0.1:5000"):
        self.server_url = server_url

    def info(self):
        info_url = ''
        api_url = f"{self.server_url}/{'model/info'}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    def detect_one(self, email_text: str):
        detect_one_url = ''
        api_url = f"{self.server_url}/{'model/email'}"
        payload = {}
        return False

    def detect_many(self, email_texts: list[str]):
        detect_many_url = ''
        api_url = f"{self.server_url}/{'model/emails'}"
        payload = {}
        return [False] * len(email_texts)


if __name__ == '__main__':
    client = SpamDetectorClient()
    print(client.info())
    print(client.detect_one("This is a test email."))
    print(client.detect_many(["This is a test email.", "Another test email."]))