"""
WebSocket server for mood detection frontend
"""
import asyncio
import websockets
import json
import cv2
import base64
import numpy as np
from main import MoodDetectionSystem

class WebSocketServer:
    def __init__(self):
        self.system = MoodDetectionSystem()
        self.clients = set()
        
    async def handle_client(self, websocket, path):
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['action'] == 'start':
                    asyncio.create_task(self.stream_detection(websocket))
                elif data['action'] == 'stop':
                    # Stop detection logic here
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def stream_detection(self, websocket):
        """Stream emotion detection results to frontend"""
        
        # Start system components
        self.system.video_capture.start()
        self.system.audio_capture.start()
        
        while True:
            try:
                # Get frame
                frame = self.system.video_capture.read_frame()
                if frame is None:
                    continue
                
                # Detect face emotion
                face, bbox = self.system.face_detector.detect_face(frame)
                face_emotion, face_conf = self.system.face_detector.predict_emotion(face)
                
                # Detect voice emotion
                audio = self.system.audio_capture.get_audio()
                voice_emotion, voice_conf = None, 0.0
                if audio is not None and len(audio) > 0:
                    voice_emotion, voice_conf = self.system.voice_detector.predict_emotion(audio)
                
                # Fuse predictions
                final_emotion, final_conf = self.system.fusion.fuse_predictions(
                    face_emotion, face_conf, voice_emotion, voice_conf
                )
                
                # Prepare data to send
                response = {
                    'face_emotion': face_emotion,
                    'face_confidence': float(face_conf) if face_conf else 0.0,
                    'voice_emotion': voice_emotion,
                    'voice_confidence': float(voice_conf) if voice_conf else 0.0,
                    'final_emotion': final_emotion,
                    'final_confidence': float(final_conf) if final_conf else 0.0
                }
                
                # Send to frontend
                await websocket.send(json.dumps(response))
                
                # Small delay
                await asyncio.sleep(0.5)
                
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"Error in stream: {e}")
                break

async def main():
    server = WebSocketServer()
    
    async with websockets.serve(server.handle_client, "localhost", 8765):
        print("🚀 WebSocket server started on ws://localhost:8765")
        print("Open the HTML file in your browser to connect!")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")


