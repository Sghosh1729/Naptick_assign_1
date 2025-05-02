import json
import random
from datetime import datetime, timedelta
import os

def dump_json(data, fname):
    os.makedirs("data", exist_ok=True)
    with open(f"data/{fname}", "w") as f:
        json.dump(data, f, indent=2)

# Simulate 1 week of wearable health data
def wearable_logs(days_back=7):
    logs = []
    for offset in range(days_back):
        dt = datetime.now() - timedelta(days=offset)
        logs.append({
            'date': dt.strftime("%Y-%m-%d"),
            'avg_heart_rate': random.randint(58, 100),
            'steps_taken': random.randint(4000, 10500),
            'sleep_duration_hrs': round(random.uniform(5.3, 8.2), 1),
            'mood': random.choice(['🙂', '😐', '😴'])
        })
    return logs

# Few previous interactions
def past_chats():
    return [
        {
            "timestamp": "2025-04-27T09:10",
            "user_msg": "Did I meet my step goal yesterday?",
            "bot_reply": "You walked 7,850 steps yesterday — just shy of your 8k goal. Almost there!"
        },
        {
            "timestamp": "2025-04-28T07:42",
            "user_msg": "How'd I sleep last night?",
            "bot_reply": "You slept 6.1 hours. Maybe try winding down earlier tonight?"
        }
    ]

# profile information
def profile():
    return {
        "name": "Shreyas Ghosh",
        "birth_year": 2002,
        "gender": "Male",
        "country": "india",
        "health_goals": ["Better REM sleep", "burn 2,500 calories per day"],
        "likes": ["morning runs", "music while working out", "photography"],
        "reminder_time": "06:30",
        "units": "metric"
    }

# GPS-stamped locations visited
def geo_trail():
    places = [
        {"place": "Riverside Trail", "time": "2025-04-26T07:15"},
        {"place": "Downtown Yoga Studio", "time": "2025-04-26T18:00"},
        {"place": "Green Bean Cafe", "time": "2025-04-27T09:45"}
    ]
    return places

# Nutrition log with quirks and casual notes
def food_notes():
    return [
        {"day": "2025-04-27", "meal": "breakfast", "cals": 390, "note": "Smoothie & toast (lazy Sunday)"},
        {"day": "2025-04-27", "meal": "lunch", "cals": 620, "note": "Poke bowl, extra rice"},
        {"day": "2025-04-27", "meal": "dinner", "cals": 710, "note": "Leftover pasta, not bad!"}
    ]

# Save all
dump_json(wearable_logs(), "wearable.json")
dump_json(past_chats(), "chat_history.json")
dump_json(profile(), "user_profile.json")
dump_json(geo_trail(), "location_data.json")
dump_json(food_notes(), "custom_collection.json")

print("datasets saved in 'data/'")
