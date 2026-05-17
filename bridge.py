import asyncio
import json
import time
import threading

import paho.mqtt.client as mqtt
import websockets

clients = set()

state = {
    "gps": {},
    "imu": {},
    "alcohol": {},
    "air": {},
    "ml": {
        "prob": 0.2,
        "probability": 0.2,
        "rash": False,
        "buf": 30,
        "model_loaded": True
    },
    "alerts": [],
    "gps_trail": [],
    "sensor_health": {
        "gps":"OK",
        "imu":"OK",
        "alcohol":"OK",
        "air":"OK",
        "gsm":"OK"
    },
    "db_rows":0,
    "uptime_s":0
}

start = time.time()

loop_ref = None

async def broadcast():

    if not clients:
        return

    msg = json.dumps({
        "type":"state",
        "data":state
    })

    dead = set()

    for ws in clients:

        try:
            await ws.send(msg)

        except:
            dead.add(ws)

    clients.difference_update(dead)

def on_message(client, userdata, msg):

    global state

    try:

        data = json.loads(
            msg.payload.decode()
        )

        print("[MQTT]", data)

        state["gps"] = {
            "lat": data["lat"],
            "lon": data["lon"],
            "speed_kmh":0,
            "hdop":1.2,
            "sats":8,
            "geofence":"none",
            "deviated":False
        }

        state["imu"] = {
            "ax": data["ax"],
            "ay": data["ay"],
            "az": data["az"],
            "gx": data["gx"],
            "gy": data["gy"],
            "gz": data["gz"]
        }

        state["alcohol"] = {
            "adc": data["alcohol"],
            "alert": data["alcohol"] > 2500
        }

        state["air"] = {
            "adc": data["air"],
            "alert": data["air"] > 3000
        }

        state["gps_trail"].append({
            "lat":data["lat"],
            "lon":data["lon"],
            "ts":time.time()
        })

        state["gps_trail"] = state["gps_trail"][-200:]

        state["db_rows"] += 1

        state["uptime_s"] = int(
            time.time()-start
        )

        asyncio.run_coroutine_threadsafe(
            broadcast(),
            loop_ref
        )

    except Exception as e:

        print(e)

async def ws_handler(websocket):

    clients.add(websocket)

    await websocket.send(json.dumps({
        "type":"state",
        "data":state
    }))

    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)

def mqtt_thread():

    client = mqtt.Client()

    client.on_message = on_message

    client.connect(
        "broker.hivemq.com",
        1883,
        60
    )

    client.subscribe("smartbus/data")

    client.loop_forever()

async def main():

    global loop_ref

    loop_ref = asyncio.get_event_loop()

    threading.Thread(
        target=mqtt_thread,
        daemon=True
    ).start()

    async with websockets.serve(
        ws_handler,
        "0.0.0.0",
        8765
    ):

        print("WS READY")

        await asyncio.Future()

asyncio.run(main())