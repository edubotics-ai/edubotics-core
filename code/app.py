from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
from chainlit.utils import mount_chainlit
import secrets
import json
import base64
from modules.config.constants import OAUTH_GOOGLE_CLIENT_ID, OAUTH_GOOGLE_CLIENT_SECRET
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

GOOGLE_CLIENT_ID = OAUTH_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = OAUTH_GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = "http://localhost:8000/auth/oauth/google/callback"

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with appropriate origins
    allow_methods=["*"],
    allow_headers=["*"],  # or specify the headers you want to allow
    expose_headers=["X-User-Info"],  # Expose the custom header
)

templates = Jinja2Templates(directory="templates")
session_store = {}
CHAINLIT_PATH = "/chainlit_tutor"

USER_ROLES = {
    "tgardos@bu.edu": ["instructor", "bu"],
    "xthomas@bu.edu": ["instructor", "bu"],
    "faridkar@bu.edu": ["instructor", "bu"],
    "xavierohan1@gmail.com": ["guest"],
    # Add more users and roles as needed
}

# Create a Google OAuth flow
flow = Flow.from_client_config(
    {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
            "scopes": [
                "openid",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
        }
    },
    scopes=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ],
    redirect_uri=GOOGLE_REDIRECT_URI,
)


def get_user_role(username: str):
    return USER_ROLES.get(username, ["student"])  # Default to "student" role


def get_user_info_from_cookie(request: Request):
    user_info_encoded = request.cookies.get("X-User-Info")
    if user_info_encoded:
        try:
            user_info_json = base64.b64decode(user_info_encoded).decode()
            return json.loads(user_info_json)
        except Exception as e:
            print(f"Error decoding user info: {e}")
            return None
    return None


def get_user_info(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token and session_token in session_store:
        return session_store[session_token]
    return None


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    user_info = get_user_info_from_cookie(request)
    if user_info and user_info.get("google_signed_in"):
        return RedirectResponse("/post-signin")
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login/guest")
@app.post("/login/guest")
async def login_guest():
    username = "guest"
    session_token = secrets.token_hex(16)
    unique_session_id = secrets.token_hex(8)
    username = f"{username}_{unique_session_id}"
    session_store[session_token] = {
        "email": username,
        "name": "Guest",
        "profile_image": "",
        "google_signed_in": False,  # Ensure guest users do not have this flag
    }
    user_info_json = json.dumps(session_store[session_token])
    user_info_encoded = base64.b64encode(user_info_json.encode()).decode()

    # Set cookies
    response = RedirectResponse(url="/post-signin", status_code=303)
    response.set_cookie(key="session_token", value=session_token)
    response.set_cookie(key="X-User-Info", value=user_info_encoded, httponly=True)
    return response


@app.get("/login/google")
async def login_google(request: Request):
    # Clear any existing session cookies to avoid conflicts with guest sessions
    response = RedirectResponse(url="/post-signin")
    response.delete_cookie(key="session_token")
    response.delete_cookie(key="X-User-Info")

    user_info = get_user_info_from_cookie(request)
    print(f"User info: {user_info}")
    # Check if user is already signed in using Google
    if user_info and user_info.get("google_signed_in"):
        return RedirectResponse("/post-signin")
    else:
        authorization_url, _ = flow.authorization_url(prompt="consent")
        return RedirectResponse(authorization_url, headers=response.headers)


@app.get("/auth/oauth/google/callback")
async def auth_google(request: Request):
    try:
        flow.fetch_token(code=request.query_params.get("code"))
        credentials = flow.credentials
        user_info = id_token.verify_oauth2_token(
            credentials.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        email = user_info["email"]
        name = user_info.get("name", "")
        profile_image = user_info.get("picture", "")

        session_token = secrets.token_hex(16)
        session_store[session_token] = {
            "email": email,
            "name": name,
            "profile_image": profile_image,
            "google_signed_in": True,  # Set this flag to True for Google-signed users
        }

        user_info_json = json.dumps(session_store[session_token])
        user_info_encoded = base64.b64encode(user_info_json.encode()).decode()

        # Set cookies
        response = RedirectResponse(url="/post-signin", status_code=303)
        response.set_cookie(key="session_token", value=session_token)
        response.set_cookie(key="X-User-Info", value=user_info_encoded, httponly=True)
        return response
    except Exception as e:
        print(f"Error during Google OAuth callback: {e}")
        return RedirectResponse(url="/", status_code=302)


@app.get("/post-signin", response_class=HTMLResponse)
async def post_signin(request: Request):
    user_info = get_user_info_from_cookie(request)
    if not user_info:
        user_info = get_user_info(request)
    if user_info and user_info.get("google_signed_in"):
        username = user_info["email"]
        role = get_user_role(username)
        jwt_token = request.cookies.get("X-User-Info")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "username": username,
                "role": role,
                "jwt_token": jwt_token,
            },
        )
    return RedirectResponse("/")


@app.post("/start-tutor")
async def start_tutor(request: Request):
    user_info = get_user_info_from_cookie(request)
    if user_info:
        user_info_json = json.dumps(user_info)
        user_info_encoded = base64.b64encode(user_info_json.encode()).decode()

        response = RedirectResponse(CHAINLIT_PATH, status_code=303)
        response.set_cookie(key="X-User-Info", value=user_info_encoded, httponly=True)
        return response

    return RedirectResponse(url="/")


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse(
        "error.html", {"request": request, "error": str(exc)}, status_code=500
    )


@app.get("/chainlit_tutor/logout", response_class=HTMLResponse)
@app.post("/chainlit_tutor/logout", response_class=HTMLResponse)
async def app_logout(request: Request, response: Response):
    # Clear session cookies
    response.delete_cookie("session_token")
    response.delete_cookie("X-User-Info")

    print("logout_page called")

    # Redirect to the logout page with embedded JavaScript
    return RedirectResponse(url="/", status_code=302)


mount_chainlit(app=app, target="main.py", path=CHAINLIT_PATH)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
