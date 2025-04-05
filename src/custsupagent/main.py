from fastapi import FastAPI, Request
from pydantic import BaseModel
from custsupagent.crew1 import RestaurantCrew
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
Crew = RestaurantCrew()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or "*" for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = Crew.crew().kickoff(inputs={"question": request.question})
    return {"response": response.raw}