import asyncio
from datetime import datetime, timedelta
import logging
import os
import time
from typing import Annotated
import uuid

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import jwt
from fastapi.security import OAuth2PasswordBearer

load_dotenv()

app = FastAPI()
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

TEST_USERNAME = "test"
TEST_PASSWORD = "test"
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_EXPIRATION_HOURS = 24  # Token expires after 24 hours

logger = logging.getLogger(__name__)

# rate limit store is mapping from user email to list of timestamps of requests
rate_limit_store: dict[str, list[float]] = {}
# connections store is mapping from user email to number of connections
connections_store: dict[str, int] = {}
# locks to protect shared data structures from race conditions
rate_limit_lock = asyncio.Lock()
connections_lock = asyncio.Lock()
RATE_LIMIT_WINDOW = 60 # 1 minute
RATE_LIMIT_MAX_REQUESTS = 3
MAX_CONNECTIONS_PER_USER = 2
MAX_GLOBAL_CONNECTIONS = 4

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
    logger.info(f"process_time: {process_time}")
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
    # now return a jwt bearer token with expiration
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    token = jwt.encode(
        {
            "email": request.email,
            "exp": expiration,
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"token": token, "expires_at": expiration.isoformat()}

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def check_rate_limit(current_user: Annotated[dict, Depends(get_current_user)]):
    user_email = current_user["email"]
    current_time = time.time()
    
    async with rate_limit_lock:
        if user_email not in rate_limit_store:
            rate_limit_store[user_email] = []
        
        # Filter old timestamps
        rate_limit_store[user_email] = [
            timestamp for timestamp in rate_limit_store[user_email] 
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
        
        # Check limit
        if len(rate_limit_store[user_email]) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                detail="Rate limit exceeded. Maximum 3 requests per minute."
            )
        
        # Append new timestamp
        rate_limit_store[user_email].append(current_time)
    
    return current_user

async def decrement_connection(user_email: str):
    async with connections_lock:
        if user_email in connections_store:
            connections_store[user_email] -= 1
            if connections_store[user_email] <= 0:
                del connections_store[user_email]

async def check_connection_limit(current_user: Annotated[dict, Depends(get_current_user)]):
    user_email = current_user["email"]
    
    async with connections_lock:
        if user_email not in connections_store:
            connections_store[user_email] = 0
        
        if connections_store[user_email] >= MAX_CONNECTIONS_PER_USER:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                detail=f"Connection limit exceeded. Maximum {MAX_CONNECTIONS_PER_USER} connections per user."
            )
        
        # Increment immediately while we hold the lock
        connections_store[user_email] += 1
    
    return current_user

async def stream_generator(prompt: str, request: Request, user_email: str):
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for event in stream:
            if await request.is_disconnected():
                break
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content
    finally:
        await decrement_connection(user_email)

@app.post("/completion")
async def completion(
    request_body: CompletionRequest,
    request: Request,
    current_user: Annotated[dict, Depends(check_connection_limit)],
):
    user_email = current_user["email"]
    return StreamingResponse(
        stream_generator(request_body.prompt, request, user_email), 
        media_type="text/event-stream"
    )


# implementing a background task that will run every 10 minutes
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_task())

async def background_task():
    while True:
        async with rate_limit_lock:
            users_to_delete = []
            current_time = time.time()
            
            # Use list() to avoid "dictionary changed size during iteration" errors
            for user_email, timestamps in list(rate_limit_store.items()):
                rate_limit_store[user_email] = [
                    timestamp for timestamp in timestamps 
                    if current_time - timestamp < RATE_LIMIT_WINDOW
                ]
                if len(rate_limit_store[user_email]) == 0:
                    users_to_delete.append(user_email)
            
            for user_email in users_to_delete:
                del rate_limit_store[user_email]
        
        await asyncio.sleep(10 * 60)