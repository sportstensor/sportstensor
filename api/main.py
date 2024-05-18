from fastapi import FastAPI
import api.db as db

app = FastAPI()

@app.get("/matches")
def read_matches():
    match_list = db.get_matches()
    if match_list:
        return {"matches": match_list}
    else:
        return {"error": "Failed to retrieve match data."}

