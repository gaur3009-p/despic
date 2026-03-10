import asyncio
import json

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

pcs = set()

async def offer(request):

    params = await request.json()

    pc = RTCPeerConnection()
    pcs.add(pc)

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=params["sdp"],
            type=params["type"]
        )
    )

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)

    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    )


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def run_server():

    app = web.Application()

    app.on_shutdown.append(on_shutdown)

    app.router.add_post("/offer", offer)

    web.run_app(app, port=8080)


if __name__ == "__main__":
    run_server()
