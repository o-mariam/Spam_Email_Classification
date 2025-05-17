import requests
import json

class SpamDetectorClient:

    def test_predict_spam():
        url = "http://127.0.0.1:5000/model/email"

        headers = {"Content-Type": "application/json; charset=utf-8"}

        data = {
            "email_text": "o dimi einai malakas",
        }

        response = requests.post(url, headers=headers, json=data)

        print("Status Code", response.status_code)
        print("JSON Response ", response.json())

        assert response.status_code==200, "Status code not ) "
        assert response.json()=={'class': 'spam'}

    def test_predict_not_spam():
        url = "http://127.0.0.1:5000/model/email"

        headers = {"Content-Type": "application/json; charset=utf-8"}

        data = {
            "email_text": "jm URL justin mason writes except for NUMBER thing defanged mime messages that s a big problem but if you didn t just remove the headers and instead reverted back to the x spam prev versions it d more or less work btw fixed the downloads page check now un defangs mime it was screwing up some of my mass check results where sa markup was present yes if there ever was a warning about sa markup in mass check it never worked for me dan ",
        }

        response = requests.post(url, headers=headers, json=data)

        print("Status Code", response.status_code)
        print("JSON Response ", response.json())

        assert response.status_code==200, "Status code not ) "
        assert response.json()=={'class': 'not_spam'}


    def test_predict_emails_not_spam():
        url = "http://127.0.0.1:5000/model/emails"

        headers = {"Content-Type": "application/json; charset=utf-8"}

        data = {
            "email_texts": ["jm URL justin mason writes except for NUMBER thing defanged mime messages that s a big problem but if you didn t just remove the headers and instead reverted back to the x spam prev versions it d more or less work btw fixed the downloads page check now un defangs mime it was screwing up some of my mass check results where sa markup was present yes if there ever was a warning about sa markup in mass check it never worked for me dan ","Adevrtise by this with discount"]
        }

        response = requests.post(url, headers=headers, json=data)

        print("Status Code", response.status_code)
        print("JSON Response ", response.json())

        assert response.status_code==200, "Status code not ) "
        assert response.json()=={'classes':["not_spam","not_spam"]}

    if __name__=="__main__":
        test_predict_emails_not_spam()
        test_predict_spam()
        test_predict_not_spam()