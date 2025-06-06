import requests
import json

import client

class SpamDetectorClientNeural(client.SpamDetectorClient):
    def __init__(self):
        super().__init__("http://127.0.0.1:5000")

if __name__ == '__main__':
    client = SpamDetectorClientNeural()
    print(client.info())
    print(client.detect_one("This is a test email."))
    print(client.detect_many(["This is a test email.", "Another test email."]))