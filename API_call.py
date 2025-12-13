import requests

def lookup_events(event_ids):
    for event_id in event_ids:
        api_call = requests.get(f"https://www.thesportsdb.com/api/v1/json/123/lookupevent.php?id={event_id}")
        storage = api_call.json()
        for event in storage["events"]:
            date_event = event["dateEvent"]
            home_team = event["strHomeTeam"]
            away_team = event["strAwayTeam"]

        print(f"{date_event}: {home_team} vs {away_team}")

event_ids = [2052711, 2052712, 2052713, 2052714]

lookup_events(event_ids)

import requests

url = "https://www.thesportsdb.com/api/v1/json/123/eventsseason.php?id=4328"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)