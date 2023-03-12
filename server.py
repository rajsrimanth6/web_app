from fastapi import FastAPI, Path, Query, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import Optional
from cam import Videocamera
from fastapi.templating import Jinja2Templates
import uvicorn
app = FastAPI()

template = Jinja2Templates(directory="")


videoCameraInstance = Videocamera()


@app.get("/")    # end point executes the function one which is below it
def home():
    return "welcome home"


@app.get("/html")
def html(request: Request):
    return template.TemplateResponse("index.html", {"request": request})


@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(gen(videoCameraInstance),
                             media_type='multipart/x-mixed-replace; boundary=frame')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        if not frame:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def release(self):
#     self.video.release()


@app.get('/stop_button', response_class=HTMLResponse)
def stop_button(request: Request):
    videoCameraInstance.release()
    return "released"


if __name__ == "__main__":
    print('stop: ctrl+c')
    uvicorn.run(app, host="0.0.0.0", port=8000)
