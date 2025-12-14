import asyncio
from datetime import timedelta
import os
import time
from typing import Annotated
import uuid

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import jwt
from fastapi.security import OAuth2PasswordBearer

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

TEST_USERNAME = "test"
TEST_PASSWORD = "test"
JWT_SECRET = os.getenv("JWT_SECRET")

# rate limit store is mapping from user email to list of timestamps of requests
rate_limit_store: dict[str, list[float]] = {}
# connections store is mapping from user email to number of connections
connections_store: dict[str, int] = {}
RATE_LIMIT_WINDOW = 60 # 1 minute
RATE_LIMIT_MAX_REQUESTS = 3
MAX_CONNECTIONS_PER_USER = 1
MAX_GLOBAL_CONNECTIONS = 2

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class CompletionRequest(BaseModel):
    prompt: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.get("/check-health")
def check_health():
    return {"message": "Hello, World!"}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print("process_time", process_time)
    response.headers["X-Process-Time"] = str(process_time)
    return response


# login to support bearer authentication
@app.post("/login")
async def login(request: LoginRequest):
    if request.email != TEST_USERNAME or request.password != TEST_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # now return a jwt bearer token
    # encode the jwt token
    token = jwt.encode(
        {
            "email": request.email,
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"token": token}

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def stream_generator(prompt: str, request: Request, user_email: str):
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for event in stream:
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content
    finally:
        await decrement_connection(user_email)

async def check_rate_limit(current_user: Annotated[dict, Depends(get_current_user)]):
    user_email = current_user["email"]
    current_time = time.time()
    if user_email not in rate_limit_store:
        rate_limit_store[user_email] = []
    rate_limit_store[user_email] = [timestamp for timestamp in rate_limit_store[user_email] if current_time - timestamp < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[user_email]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Maximum 3 requests per minute.")
    rate_limit_store[user_email].append(current_time)
    return current_user

async def increment_connection(user_email: str):
    if user_email not in connections_store:
        connections_store[user_email] = 0
    connections_store[user_email] += 1

async def decrement_connection(user_email: str):
    if user_email in connections_store:
        connections_store[user_email] -= 1
        if connections_store[user_email] <= 0:
            del connections_store[user_email]

async def check_connection_limit(current_user: Annotated[dict, Depends(get_current_user)]):
    user_email = current_user["email"]
    if user_email not in connections_store:
        connections_store[user_email] = 0
    if connections_store[user_email] >= MAX_CONNECTIONS_PER_USER:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Connection limit exceeded. Maximum {MAX_CONNECTIONS_PER_USER} connections per user.")
    return current_user
            
@app.post("/completion")
async def completion(
    request_body: CompletionRequest,
    request: Request,
    current_user: Annotated[dict, Depends(check_connection_limit)],
):
    user_email = current_user["email"]
    await increment_connection(user_email)
    return StreamingResponse(stream_generator(request_body.prompt, request, user_email), media_type="text/event-stream")


# implementing a background task that will run every 10 minutes
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_task())

async def background_task():
    while True:
        users_to_delete = []
        current_time = time.time()
        for user_email, timestamps in rate_limit_store.items():
            rate_limit_store[user_email] = [timestamp for timestamp in timestamps if current_time - timestamp < RATE_LIMIT_WINDOW]
            if len(rate_limit_store[user_email]) == 0:
                users_to_delete.append(user_email)
        for user_email in users_to_delete:
            del rate_limit_store[user_email]
        await asyncio.sleep(10 * 60)