import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

pcs = set()

async def offer(request):

    params = await request.json()

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    async def on_track(track):

        if track.kind == "audio":
            print("Receiving audio stream")

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


async def index(request):

    return web.FileResponse("app/realtime/index.html")


async def on_shutdown(app):

    coros = [pc.close() for pc in pcs]

    await asyncio.gather(*coros)

    pcs.clear()


def run():

    app = web.Application()

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    app.on_shutdown.append(on_shutdown)
    web.run_app(app, port=7860)

if __name__ == "__main__":
    run()
