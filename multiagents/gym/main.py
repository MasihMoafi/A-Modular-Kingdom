from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from gym_crew import GymCrew
from database import Database

app = FastAPI(title="AI Gym Assistant")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database and crew
db = Database()
gym_crew = GymCrew()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-interview")
async def start_interview(request: Request, user_id: str = Form(...)):
    # Start interview process
    result = gym_crew.start_user_journey(user_id)
    return {"status": "success", "message": result}

@app.post("/submit-response")
async def submit_response(request: Request, user_id: str = Form(...), response: str = Form(...)):
    # Process user response
    result = gym_crew.process_user_response(user_id, response)
    return {"status": "success", "data": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 