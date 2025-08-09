import websocket
import json
def test_ngrok_websocket():
    try:
        ws = websocket.create_connection("wss://889cab5850f5.ngrok-free.app/mcp")
        print("ngrok success")
        ws.close()
    except Exception as e:
        print(f"ngrok {e}")

