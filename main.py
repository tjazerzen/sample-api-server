import asyncio
from datetime import datetime, timedelta
import io
import logging
import os
import time
from typing import BinaryIO, Annotated

from fastapi import FastAPI, HTTPException, Depends, Request, status, File, UploadFile
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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class MetricsStore:
    def __init__(self):
        self.total_requests = 0
        self.endpoint_requests = {}  # {endpoint: count}
        self.endpoint_errors = {}    # {endpoint: {error_type: count}}
        self.total_tokens_processed = 0
        self.request_latencies = []  # Store recent latencies (keep last 1000)
        self.max_latency_samples = 1000
        self.lock = asyncio.Lock()
    
    async def increment_request(self, endpoint: str):
        async with self.lock:
            self.total_requests += 1
            self.endpoint_requests[endpoint] = self.endpoint_requests.get(endpoint, 0) + 1
    
    async def increment_error(self, endpoint: str, error_type: str):
        async with self.lock:
            if endpoint not in self.endpoint_errors:
                self.endpoint_errors[endpoint] = {}
            self.endpoint_errors[endpoint][error_type] = \
                self.endpoint_errors[endpoint].get(error_type, 0) + 1
    
    async def add_tokens(self, count: int):
        async with self.lock:
            self.total_tokens_processed += count
    
    async def record_latency(self, endpoint: str, latency_seconds: float):
        async with self.lock:
            self.request_latencies.append({
                "endpoint": endpoint,
                "latency": latency_seconds,
                "timestamp": time.time()
            })
            # Keep only recent samples
            if len(self.request_latencies) > self.max_latency_samples:
                self.request_latencies = self.request_latencies[-self.max_latency_samples:]
    
    async def get_active_streams(self) -> int:
        """Get current number of active streams from connections_store"""
        async with connections_lock:
            return sum(connections_store.values())
    
    async def get_metrics_summary(self):
        async with self.lock:
            # Calculate latency statistics
            latency_stats = {}
            if self.request_latencies:
                latencies = [r["latency"] for r in self.request_latencies]
                latency_stats = {
                    "avg": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "count": len(latencies)
                }
            
            # Calculate per-endpoint latency averages
            endpoint_latencies = {}
            for record in self.request_latencies:
                ep = record["endpoint"]
                if ep not in endpoint_latencies:
                    endpoint_latencies[ep] = []
                endpoint_latencies[ep].append(record["latency"])
            
            endpoint_latency_avg = {
                ep: sum(lats) / len(lats) 
                for ep, lats in endpoint_latencies.items()
            }
            
            return {
                "total_requests": self.total_requests,
                "endpoint_requests": self.endpoint_requests,
                "endpoint_errors": self.endpoint_errors,
                "total_tokens_processed": self.total_tokens_processed,
                "active_streams": await self.get_active_streams(),
                "latency_stats": latency_stats,
                "endpoint_latency_avg": endpoint_latency_avg
            }

# Initialize metrics store
metrics = MetricsStore()

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
    
    # Get the endpoint path
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record metrics
        await metrics.increment_request(endpoint)
        await metrics.record_latency(endpoint, process_time)
        
        logger.info(f"process_time: {process_time}")
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        
        # Record error metrics
        error_type = type(e).__name__
        await metrics.increment_error(endpoint, error_type)
        await metrics.record_latency(endpoint, process_time)
        
        # Re-raise the exception to let FastAPI handle it
        raise


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

async def stream_text_completion_generator(prompt: str, request: Request, user_email: str):
    total_chars = 0  # Simple approximation: 1 token ≈ 4 chars
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
                content = event.choices[0].delta.content
                total_chars += len(content)
                yield f"data: {content}\n\n"
    finally:
        # Approximate token count (rough: 4 chars per token)
        estimated_tokens = total_chars // 4
        await metrics.add_tokens(estimated_tokens)
        await decrement_connection(user_email)

@app.post("/completion")
async def completion(
    request_body: CompletionRequest,
    request: Request,
    current_user: Annotated[dict, Depends(check_connection_limit)],
):
    user_email = current_user["email"]
    return StreamingResponse(
        stream_text_completion_generator(request_body.prompt, request, user_email), 
        media_type="text/event-stream"
    )

async def stream_text_to_speech_generator(audio_file: BinaryIO):
    final_text = ""
    stream = await client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
        response_format="text",
        stream=True,
    )

    async for event in stream:
        if event.type == "transcript.text.delta":
            final_text += event.delta
            # yield f"data: {event.delta}\n\n"
            yield f"data: {str(event)}\n\n"
            await asyncio.sleep(0.1)

@app.post("/completion-stt")
async def completion_stt(
    audio_file: UploadFile = File(...),
    request: Request = None,
    current_user: Annotated[dict, Depends(check_connection_limit)] = None,
):
    user_email = current_user["email"]
    audio_file = open(audio_file.filename, "rb")
    return StreamingResponse(
        stream_text_to_speech_generator(audio_file),
        media_type="text/event-stream"
    )

@app.get("/metrics")
async def get_metrics():
    """Expose current metrics"""
    return await metrics.get_metrics_summary()

@app.get("/metrics/dashboard")
async def metrics_dashboard():
    """Human-readable metrics dashboard"""
    summary = await metrics.get_metrics_summary()
    
    # Format for readability
    dashboard = {
        "overview": {
            "total_requests": summary["total_requests"],
            "active_streams": summary["active_streams"],
            "total_tokens_processed": summary["total_tokens_processed"],
        },
        "requests_by_endpoint": summary["endpoint_requests"],
        "errors_by_endpoint": summary["endpoint_errors"],
        "latency": {
            "overall": summary["latency_stats"],
            "by_endpoint": summary["endpoint_latency_avg"]
        }
    }
    return dashboard


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