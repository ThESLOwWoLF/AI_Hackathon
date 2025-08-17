import os
import json
import logging
from typing import Any, Dict
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from mcp.server import Server, NotificationOptions, InitializationOptions
from mcp.types import Tool, CallToolResult, TextContent
from datetime import datetime, timedelta
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting storage
class RateLimiter:
    def __init__(self, max_requests: int = 50, window_hours: int = 24):
        self.max_requests = max_requests
        self.window_hours = window_hours
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, ip: str) -> bool:
        try:
            now = datetime.now()
            cutoff = now - timedelta(hours=self.window_hours)
            
            with self.lock:
                # Clean old requests
                self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > cutoff]
                
                # Check if under limit
                if len(self.requests[ip]) < self.max_requests:
                    self.requests[ip].append(now)
                    return True
                
                return False
        except Exception as e:
            logger.error(f"Rate limit error for IP {ip}: {e}")
            return True
    
    def get_remaining_requests(self, ip: str) -> int:
        try:
            now = datetime.now()
            cutoff = now - timedelta(hours=self.window_hours)
            
            with self.lock:
                # Clean old requests
                self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > cutoff]
                
                return max(0, self.max_requests - len(self.requests[ip]))
        except Exception as e:
            logger.error(f"Error getting remaining requests for IP {ip}: {e}")
            return self.max_requests

# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=50, window_hours=24)

# Rate limiting dependency
async def check_rate_limit(request: Request):
    """Dependency to check rate limits for each request"""
    client_ip = request.client.host
    
    # Handle cases where client IP might be None (e.g., from proxy)
    if not client_ip:
        client_ip = "unknown"
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        remaining_time = rate_limiter.window_hours
        raise HTTPException(
            status_code=429, 
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum {rate_limiter.max_requests} requests per {rate_limiter.window_hours} hours exceeded",
                "limit_reset": f"Resets in {remaining_time} hours",
                "ip": client_ip
            }
        )
    
    # Return client IP for use in the endpoint
    return client_ip

def add_rate_limit_headers(response: Response, client_ip: str):
    """Add rate limit headers to response"""
    remaining = rate_limiter.get_remaining_requests(client_ip)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    response.headers["X-RateLimit-Reset"] = str(rate_limiter.window_hours)
    return response

server = Server("puch_ai")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "letmein")
MOCK_USERS = {
    "RochitSudhan#1802": "+91 8197082621"
}

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCo_9pVuEiuCK2r-7R1ztttv2W1pSFBaDE")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to configure Gemini: {e}")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(
        name="validate",
        description="Validate bearer token and return user's phone number for Puch AI authentication",
        inputSchema={
            "type": "object",
            "properties": {"bearer_token": {"type": "string", "description": "Bearer token for authentication"}},
            "required": ["bearer_token"]
        }
    ), Tool(
        name="ai_trip_planner",
        description="AI-powered trip planner that creates personalized travel itineraries based on destination, budget, duration, and preferences",
        inputSchema={
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "Destination city/country for the trip"},
                "duration": {"type": "string", "description": "Duration of the trip (e.g., '3 days', '1 week')"},
                "budget": {"type": "string", "description": "Budget range (e.g., 'budget', 'moderate', 'luxury')"},
                "interests": {"type": "string", "description": "Interests and activities (e.g., 'culture, food, adventure')"},
                "travel_style": {"type": "string", "description": "Travel style preference (e.g., 'relaxed', 'fast-paced', 'family-friendly')"}
            },
            "required": ["destination", "duration"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    if name == "validate":
        bearer_token = arguments.get("bearer_token", "")
        if bearer_token in MOCK_USERS:
            phone_number = MOCK_USERS[bearer_token]
            return CallToolResult(content=[TextContent(type="text", text=phone_number)])
        else:
            return CallToolResult(content=[TextContent(type="text", text="Invalid bearer token")], isError=True)
    elif name == "ai_trip_planner":
        destination = str(arguments.get("destination", "")).strip()
        duration = str(arguments.get("duration", "")).strip()
        budget = str(arguments.get("budget", "moderate")).strip()
        interests = str(arguments.get("interests", "general")).strip()
        travel_style = str(arguments.get("travel_style", "balanced")).strip()

        if not GEMINI_API_KEY:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=(
                        "Gemini API key not configured. Set 'GEMINI_API_KEY' in the environment "
                        "to enable the AI Trip Planner."
                    ),
                )],
                isError=True,
            )

        def build_prompt() -> str:
            return (
                "You are an expert travel planner. Based on the inputs, return ONLY valid JSON with the following keys: "
                "famous_places (array of objects with name and short_description), "
                "best_time (string), local_food (array of strings), culture_tips (string), "
                "itinerary (array of objects with day (int starting at 1), title (string), activities (array of strings)). "
                "Avoid markdown and extra commentary. "
                f"Destination: {destination}. Duration: {duration}. Budget: {budget}. Interests: {interests}. Travel style: {travel_style}. "
                "Ensure at least 8-12 famous_places populated for popular cities; when unsure, still return reasonable suggestions."
            )

        def extract_json(text: str) -> dict:
            t = (text or "").strip()
            # Strip markdown code fences if present
            if t.startswith("```"):
                t = t.strip('`')
                # Remove leading language tag like ```json
                first_newline = t.find("\n")
                if first_newline != -1:
                    t = t[first_newline+1:]
            # Remove trailing code fence if left
            if t.endswith("```"):
                t = t[:-3]
            return json.loads(t)

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(build_prompt())
            raw_text = (getattr(response, "text", None) or "").strip()
            data = extract_json(raw_text)
        except Exception as e:
            logger.exception("Gemini trip planner failed")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Trip planning failed: {str(e)}")],
                isError=True,
            )

        famous_places = data.get("famous_places", [])
        best_time = data.get("best_time", "")
        local_food = data.get("local_food", [])
        culture_tips = data.get("culture_tips", "")
        itinerary_days = data.get("itinerary", [])

        def format_places(items: list[dict]) -> str:
            lines = []
            for place in items[:12]:
                name = place.get("name") if isinstance(place, dict) else str(place)
                desc = place.get("short_description", "") if isinstance(place, dict) else ""
                if desc:
                    lines.append(f"• {name} — {desc}")
                else:
                    lines.append(f"• {name}")
            return "\n".join(lines) if lines else "• (no data)"

        def format_itinerary(items: list[dict]) -> str:
            parts = []
            for day in items:
                day_num = day.get("day")
                title = day.get("title", f"Day {day_num or ''}").strip()
                activities = day.get("activities", [])
                parts.append(
                    "\n".join([
                        f"**Day {day_num or len(parts)+1}: {title}**",
                        *(f"• {act}" for act in activities[:8]),
                        "",
                    ])
                )
            return "\n".join(parts) if parts else "**Day 1:** • Explore city center"

        output = (
            f"🌍 AI Trip Planner — {destination}\n\n"
            f"**Trip Details**\n"
            f"• Destination: {destination}\n"
            f"• Duration: {duration}\n"
            f"• Budget: {budget}\n"
            f"• Interests: {interests}\n"
            f"• Travel Style: {travel_style}\n\n"
            f"**Famous Places**\n{format_places(famous_places)}\n\n"
            f"**Best Time To Visit**\n• {best_time}\n\n"
            f"**Local Food**\n{chr(10).join(f'• {x}' for x in local_food[:10]) or '• (no data)'}\n\n"
            f"**Culture Tips**\n• {culture_tips or '(no data)'}\n\n"
            f"**Itinerary**\n{format_itinerary(itinerary_days)}\n"
        )

        return CallToolResult(content=[TextContent(type="text", text=output)])
    
    return CallToolResult(content=[TextContent(type="text", text=f"Tool not found: {name}")], isError=True)

app = FastAPI(title="Puch AI MCP Server", version="1.0.0")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Puch AI MCP Server started with rate limiting: {rate_limiter.max_requests} requests per {rate_limiter.window_hours} hours per IP")

# Root endpoint
@app.get("/")
@app.post("/")
async def read_root(client_ip: str = Depends(check_rate_limit)):
    response_data = {"message": "Puch AI MCP Server | Built by Sudhan and Rochit"}
    response = JSONResponse(content=response_data)
    add_rate_limit_headers(response, client_ip)
    return response

# MCP HTTP endpoint
@app.post("/mcp")
async def mcp_http_endpoint(request: Request, client_ip: str = Depends(check_rate_limit)):
    body = await request.body()
    try:
        data = json.loads(body.decode())
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")

        if method == "initialize":
            response = JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "PuchAI", "version": "1.0.0"}
                }
            })
            add_rate_limit_headers(response, client_ip)
            return response
        elif method == "notifications/initialized":
            response = Response(status_code=200)
            add_rate_limit_headers(response, client_ip)
            return response
        elif method == "tools/list":
            tools = await list_tools()
            response = JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]
                }
            })
            add_rate_limit_headers(response, client_ip)
            return response
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                arguments["bearer_token"] = auth_header[7:]
            
            result = await call_tool(name, arguments)
            response = JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": result.content[0].text}],
                    "isError": getattr(result, 'isError', False)
                }
            })
            add_rate_limit_headers(response, client_ip)
            return response
        
        response = JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}, status_code=400)
        add_rate_limit_headers(response, client_ip)
        return response
    except Exception as e:
        response = JSONResponse({"jsonrpc": "2.0", "id": data.get("id") if 'data' in locals() else None, "error": {"code": -32603, "message": str(e)}}, status_code=500)
        add_rate_limit_headers(response, client_ip)
        return response

# WebSocket endpoint
@app.websocket("/mcp")
async def mcp_websocket_endpoint(websocket: WebSocket):
    try:
        # Get client IP from WebSocket
        client_ip = websocket.client.host
        if not client_ip:
            client_ip = "unknown"
        
        # Check rate limit before accepting connection
        if not rate_limiter.is_allowed(client_ip):
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return
        
        await websocket.accept()
        
        await server.run(websocket.receive_text, websocket.send_text, InitializationOptions(
            server_name="puch_ai",
            server_version="1.0.0",
            capabilities=server.get_capabilities(notification_options=NotificationOptions())
        ))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
