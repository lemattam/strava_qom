import requests
import time
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

CLIENT_ID = [id]
CLIENT_SECRET = [secret]
REDIRECT_URI = 'http://localhost/exchange_token'
REQUEST_DELAY = 15

def get_access_token():

    '''gets access token for strava api from client secret and client id'''
    
    print("Go to this URL and authorize the app:")
    
    auth_url = (
        f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}"
        f"&response_type=code&redirect_uri={REDIRECT_URI}"
        f"&scope=activity:read_all&approval_prompt=force"
    )
    
    print(auth_url)

    code = input("\nPaste the code from the redirected URL:\n> ").strip()

    token_resp = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    })
    token_data = token_resp.json()

    if "access_token" not in token_data:
        raise Exception(f"Failed to get access token: {token_data}")

    return token_data["access_token"]


def get_segment_ids_from_activity(activity_id, access_token):

    '''gets all segment ids from activity ids'''
    
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    resp = requests.get(url, headers=headers, params={"include_all_efforts": True})
    time.sleep(REQUEST_DELAY)
    data = resp.json()

    if "segment_efforts" not in data:
        print(f"No segment efforts found for activity {activity_id}")
        return []

    segment_ids = [effort["segment"]["id"] for effort in data["segment_efforts"]]

    #print(f"\nSegment IDs from activity {activity_id}:")
    for sid in segment_ids:
        print(f"- {sid}")

    return segment_ids


def get_segment_efforts_and_qom(segment_id, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}

    efforts_url = "https://www.strava.com/api/v3/segment_efforts"
    efforts_response = requests.get(efforts_url, headers=headers, params={
        "segment_id": segment_id,
        "per_page": 100
    })

    time.sleep(REQUEST_DELAY)
    efforts_data = efforts_response.json()

    #print(f"{segment_id} Efforts:")

    times = []

    if isinstance(efforts_data, list) and efforts_data:
        for effort in efforts_data:
            time_ = effort['elapsed_time']
            #print(f"{effort['start_date_local']} = {time_} seconds")
            times.append(time_)

    else:
        print("No efforts found ... error?", efforts_data)

    segment_url = f"https://www.strava.com/api/v3/segments/{segment_id}"
    segment_response = requests.get(segment_url, headers=headers)

    time.sleep(REQUEST_DELAY)

    segment_data = segment_response.json()

    name = segment_data.get('name')
    distance = segment_data.get('distance')   
    grade = segment_data.get('average_grade')

    #print(f"\n--- Segment Info ---")
    #print(f"Name: {name}")

    #if times:
        #print(f"my time: {times[0]}")

    #else:
        #print("my time: No effort times available")

    #print(f"Distance: {distance} meters")
    #print(f"Average Grade: {grade}%")

    qom_effort = segment_data.get("xoms", {}).get("qom")

    #if qom_effort:
        #print(f"QOM Time: {qom_effort} seconds")
    #else:
        #print("QOM time not found.")

    alls = []

    for time_ in times:
        alls.append((name, distance, grade, time_, qom_effort))

    return alls


def get_recent_activity_ids(access_token, limit=20):

    '''gets n = limit most recent activity ids from my profile'''
    
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://www.strava.com/api/v3/athlete/activities"
    resp = requests.get(url, headers=headers, params={"per_page": limit})
    time.sleep(REQUEST_DELAY)
    data = resp.json()

    if not isinstance(data, list):
        print("didn't get activities:", data)
        return []

    activity_ids = [activity["id"] for activity in data]

    #print(f"\nYour {len(activity_ids)} most recent activity IDs:")
    
    #for aid in activity_ids:
        #print(f"- {aid}")

    return activity_ids

'''gets all segments infos from my profile's recent activities'''

if __name__ == "__main__":
    
    access_token = get_access_token()
    activity_ids = get_recent_activity_ids(access_token, limit=5)

    segment_ids = []
    for activity_id in activity_ids:
        segment_ids.extend(get_segment_ids_from_activity(activity_id, access_token))

    segment_efforts = []

    for segment_id in segment_ids:
        segment_efforts.extend(get_segment_efforts_and_qom(segment_id, access_token))
        print(segment_efforts)
        
print(segment_efforts)



###########


def time_to_seconds(t):

    '''turns strava's xom format xx:xx into seconds'''
    
    if pd.isna(t):
        return t

    t = str(t).strip().lower().replace("s", "") 
    if ":" in t:
        parts = list(map(int, t.split(":")))
        return sum(x * 60**i for i, x in enumerate(reversed(parts)))
    else:
        try:
            return int(t)
        except ValueError:
            return np.nan  


def impute(data):

    '''imputes non-numerical data wiht imputed data based on bayesian ridge regression'''
    
    cols_to_impute = ["Distance", "Grade", "My time", "QOM time"]
    
    estimator = BayesianRidge()
    imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=0)
    imputed = imputer.fit_transform(data[cols_to_impute])
    data[cols_to_impute] = imputed
    
    return data


def make_df(segment_efforts):

    '''makes df out of segment data list'''
    df = pd.DataFrame(segment_efforts, columns=["Segment Name", "Distance", "Grade", "My time", "QOM time"])
    
    df["QOM time"] = df["QOM time"].apply(lambda x: time_to_seconds(x) if pd.notna(x) else x)
    df["My time"] = df["My time"].apply(lambda x: time_to_seconds(x) if pd.notna(x) else x)

    df = impute(df)
    
    df["QOM time"] = df["QOM time"].astype(int)
    df["My time"] = df["My time"].astype(int)
    df["Ratio from QOM"] = df["My time"] / df["QOM time"]
    
    df = df.drop(columns=["My time", "QOM time"])
    
    display(df)
    
    return df


def train_ratio_model(df):
    
    '''trains random forest model, determins ratio from QOM based on distance and gradient of a segment'''

    X = df[["Distance", "Grade"]]
    y = df["Ratio from QOM"]
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    
    return model


def predict_ratio(model, distance, grade):

    '''predicts QOM likelihood of segments ... used same segments bc training pool was so small and I think there's enough nuance'''

    X_new = np.array([[distance, grade]])
    return model.predict(X_new)[0]


def main(segments):

    '''takes in segments, returns optimal ones'''

    df = make_df(segments)
    model = train_ratio_model(df)

    df["Predicted Ratio"] = df.apply(lambda row: predict_ratio(model, row["Distance"], row["Grade"]), axis=1)
    optimal_segments = df[df["Predicted Ratio"] < 1.3].sort_values("Predicted Ratio")
    
    return optimal_segments[["Segment Name", "Distance", "Grade", "Predicted Ratio"]]

    
main(segment_efforts)

