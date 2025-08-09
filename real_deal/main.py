import os
import json
import logging
from typing import Any, Dict
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, Response
from mcp.server import Server, NotificationOptions, InitializationOptions
from mcp.types import Tool, CallToolResult, TextContent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

server = Server("puch_ai")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "letmein")
MOCK_USERS = {
    "abc123token": "1234567890" # da we should add our actuall numbers here for validation
}

@server.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Tools requested")
    return [Tool(
        name="validate",
        description="Validate bearer token and return user's phone number for Puch AI authentication",
        inputSchema={
            "type": "object",
            "properties": {"bearer_token": {"type": "string", "description": "Bearer token for authentication"}},
            "required": ["bearer_token"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    if name == "validate":
        bearer_token = arguments.get("bearer_token", "")
        if bearer_token in MOCK_USERS:
            phone_number = MOCK_USERS[bearer_token]
            return CallToolResult(content=[TextContent(type="text", text=phone_number)])
        else:
            return CallToolResult(content=[TextContent(type="text", text="Invalid bearer token")], isError=True)
    return CallToolResult(content=[TextContent(type="text", text=f"Tool not found: {name}")], isError=True)

app = FastAPI()

@app.post("/")
@app.post("/mcp")
async def mcp_http_endpoint(request: Request):
    logger.info(f"HTTP POST with headers: {dict(request.headers)}")
    body = await request.body()
    try:
        data = json.loads(body.decode())
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")

        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "puch_ai", "version": "0.0.1"}
                }
            })
        elif method == "notifications/initialized":
            return Response(status_code=200)
        elif method == "tools/list":
            tools = await list_tools()
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]
                }
            })
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                arguments["bearer_token"] = auth_header[7:]
            result = await call_tool(name, arguments)
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": result.content[0].text}],
                    "isError": getattr(result, 'isError', False)
                }
            })
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}, status_code=400)
    except Exception as e:
        return JSONResponse({"jsonrpc": "2.0", "id": data.get("id") if 'data' in locals() else None, "error": {"code": -32603, "message": str(e)}}, status_code=500)

@app.websocket("/mcp")
async def mcp_websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        await server.run(websocket.receive_text, websocket.send_text, InitializationOptions(
            server_name="puch_ai",
            server_version="0.0.1",
            capabilities=server.get_capabilities(notification_options=NotificationOptions())
        ))
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
