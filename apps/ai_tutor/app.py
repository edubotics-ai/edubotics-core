from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
from chainlit.utils import mount_chainlit
import secrets
import json
import base64
from config.constants import (
    OAUTH_GOOGLE_CLIENT_ID,
    OAUTH_GOOGLE_CLIENT_SECRET,
    CHAINLIT_URL,
    EMAIL_ENCRYPTION_KEY,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from helpers import (
    get_time,
    reset_tokens_for_user,
    check_user_cooldown,
)
from edubotics_core.chat_processor.helpers import get_user_details, update_user_info
from config.config_manager import config_manager
import hashlib
from typing import Optional
from database import DatabaseFactory, Database

# Set up configuration
config = config_manager.get_config().dict()

# Set constants
GITHUB_REPO = config["misc"]["github_repo"]
DOCS_WEBSITE = config["misc"]["docs_website"]
ALL_TIME_TOKENS_ALLOCATED = config["token_config"]["all_time_tokens_allocated"]
TOKENS_LEFT = config["token_config"]["tokens_left"]
COOLDOWN_TIME = config["token_config"]["cooldown_time"]
REGEN_TIME = config["token_config"]["regen_time"]

GOOGLE_CLIENT_ID = OAUTH_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = OAUTH_GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = f"{CHAINLIT_URL}/auth/oauth/google/callback"

# Initialize FastAPI app
app = FastAPI()

# Serve static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Info"],
)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Set the Chainlit path
CHAINLIT_PATH = "/chainlit_tutor"

# Load user roles from an encrypted JSON file
with open("public/files/students_encrypted.json", "r") as file:
    USER_ROLES = json.load(file)

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
                # "https://www.googleapis.com/auth/userinfo.email",
                # "https://www.googleapis.com/auth/userinfo.profile",
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
    def deterministic_hash(email, salt):
        return hashlib.pbkdf2_hmac("sha256", email.encode(), salt, 100000).hex()

    encryption_salt = EMAIL_ENCRYPTION_KEY.encode()
    encrypted_email = deterministic_hash(username, encryption_salt)
    role = USER_ROLES.get(encrypted_email, ["guest"])

    if "guest" in role:
        return "unauthorized"

    return role


# get DB instance
def get_db() -> Database:
    db_type = config.get("database_type", "sqlite")  # Default to SQLite
    print(f"Using database type: {db_type}")
    return DatabaseFactory.create_database(db_type)


# Get user info from cookie
async def get_user_info_from_cookie(request: Request) -> Optional[dict]:
    user_info_encoded = request.cookies.get("X-User-Info")
    if user_info_encoded:
        try:
            user_info_json = base64.b64decode(user_info_encoded).decode()
            return json.loads(user_info_json)
        except Exception as e:
            print(f"Error decoding user info: {e}")
            return None
    return None


# Delete user info from cookie and session store
async def del_user_info_from_cookie(
    request: Request, response: Response, db: Database = Depends(get_db)
):
    response.delete_cookie("X-User-Info")
    response.delete_cookie("session_token")
    session_token = request.cookies.get("session_token")
    if session_token:
        session = db.create_session()
        db.delete(session, session_token)
        db.commit(session)
        db.close(session)


# Get user info from session store
def get_user_info(request: Request, db: Database = Depends(get_db)) -> Optional[dict]:
    session_token = request.cookies.get("session_token")
    if session_token:
        session = db.create_session()
        user_data = db.get(session, session_token)
        db.close(session)
        return user_data
    return None


# Login page route
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    user_info = await get_user_info_from_cookie(request)
    if user_info and user_info.get("google_signed_in"):
        return RedirectResponse("/post-signin")
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "GITHUB_REPO": GITHUB_REPO, "DOCS_WEBSITE": DOCS_WEBSITE},
    )


# Unauthorized route
@app.get("/unauthorized", response_class=HTMLResponse)
async def unauthorized(request: Request):
    return templates.TemplateResponse("unauthorized.html", {"request": request})


# Google login route
@app.get("/login/google")
async def login_google(request: Request):
    response = RedirectResponse(url="/post-signin")
    response.delete_cookie(key="session_token")
    response.delete_cookie(key="X-User-Info")

    user_info = await get_user_info_from_cookie(request)
    if user_info and user_info.get("google_signed_in"):
        return RedirectResponse("/post-signin")
    else:
        authorization_url, _ = flow.authorization_url(prompt="consent")
        return RedirectResponse(authorization_url, headers=response.headers)


# Google OAuth callback route
@app.get("/auth/oauth/google/callback")
async def auth_google(request: Request, db: Database = Depends(get_db)):
    try:
        flow.fetch_token(code=request.query_params.get("code"))
        credentials = flow.credentials
        user_info = id_token.verify_oauth2_token(
            credentials.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        email = user_info["email"]
        name = user_info.get("name", "")
        profile_image = user_info.get("picture", "")
        role = get_user_role(email)

        if role == "unauthorized":
            return RedirectResponse("/unauthorized")

        session_token = secrets.token_hex(16)
        literalai_user = await get_user_details(email)

        session_data = {
            "session_token": session_token,
            "email": email,
            "name": name,
            "profile_image": profile_image,
            "google_signed_in": True,
            "literalai_info": literalai_user.to_dict(),
        }

        session = db.create_session()
        db.add(session, session_data)
        db.commit(session)
        db.close(session)

        user_info_json = json.dumps(session_data)
        user_info_encoded = base64.b64encode(user_info_json.encode()).decode()

        response = RedirectResponse(url="/post-signin", status_code=303)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        response.set_cookie(key="X-User-Info", value=user_info_encoded, httponly=True)
        return response
    except Exception as e:
        print(f"Error during Google OAuth callback: {e}")
        return RedirectResponse(url="/", status_code=302)


# Cooldown route
@app.get("/cooldown", response_class=HTMLResponse)
async def cooldown(request: Request, db: Database = Depends(get_db)):
    user_info = await get_user_info_from_cookie(request)
    user_details = await get_user_details(user_info["email"])
    current_datetime = get_time()
    user_role = get_user_role(user_info["email"])
    cooldown, cooldown_end_time = await check_user_cooldown(
        user_details, current_datetime, COOLDOWN_TIME, TOKENS_LEFT, REGEN_TIME
    )

    if cooldown and "admin" not in user_role:
        return templates.TemplateResponse(
            "cooldown.html",
            {
                "request": request,
                "username": user_info["email"],
                "role": user_role,
                "cooldown_end_time": cooldown_end_time,
                "tokens_left": user_details.metadata["tokens_left"],
            },
        )
    else:
        user_details.metadata["in_cooldown"] = False
        await update_user_info(user_details)
        await reset_tokens_for_user(
            user_details,
            config["token_config"]["tokens_left"],
            config["token_config"]["regen_time"],
        )
        return RedirectResponse("/post-signin")


# Post-signin route
@app.get("/post-signin", response_class=HTMLResponse)
async def post_signin(request: Request, db: Database = Depends(get_db)):
    user_info = await get_user_info_from_cookie(request)
    if not user_info:
        user_info = get_user_info(request, db)
    user_details = await get_user_details(user_info["email"])
    current_datetime = get_time()
    user_role = get_user_role(user_info["email"])
    user_details.metadata["last_login"] = current_datetime

    if "tokens_left" not in user_details.metadata:
        user_details.metadata["tokens_left"] = TOKENS_LEFT
    if "last_message_time" not in user_details.metadata:
        user_details.metadata["last_message_time"] = current_datetime
    if "all_time_tokens_allocated" not in user_details.metadata:
        user_details.metadata["all_time_tokens_allocated"] = ALL_TIME_TOKENS_ALLOCATED
    if "in_cooldown" not in user_details.metadata:
        user_details.metadata["in_cooldown"] = False

    await update_user_info(user_details)

    if "last_message_time" in user_details.metadata and "admin" not in user_role:
        cooldown, _ = await check_user_cooldown(
            user_details, current_datetime, COOLDOWN_TIME, TOKENS_LEFT, REGEN_TIME
        )
        if cooldown:
            user_details.metadata["in_cooldown"] = True
            return RedirectResponse("/cooldown")
        else:
            user_details.metadata["in_cooldown"] = False
            await reset_tokens_for_user(
                user_details,
                config["token_config"]["tokens_left"],
                config["token_config"]["regen_time"],
            )

    if user_info:
        username = user_info["email"]
        jwt_token = request.cookies.get("X-User-Info")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "username": username,
                "role": user_role,
                "jwt_token": jwt_token,
                "tokens_left": user_details.metadata["tokens_left"],
                "all_time_tokens_allocated": user_details.metadata[
                    "all_time_tokens_allocated"
                ],
                "total_tokens_allocated": ALL_TIME_TOKENS_ALLOCATED,
            },
        )
    return RedirectResponse("/")


# Start tutor route
@app.get("/start-tutor")
@app.post("/start-tutor")
async def start_tutor(request: Request, db: Database = Depends(get_db)):
    user_info = await get_user_info_from_cookie(request)
    if user_info:
        user_info_json = json.dumps(user_info)
        user_info_encoded = base64.b64encode(user_info_json.encode()).decode()

        response = RedirectResponse(CHAINLIT_PATH, status_code=303)
        response.set_cookie(key="X-User-Info", value=user_info_encoded, httponly=True)
        return response

    return RedirectResponse(url="/")


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse(
            "error_404.html", {"request": request}, status_code=404
        )
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error": str(exc)},
        status_code=exc.status_code,
    )


# General exception handler
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse(
        "error.html", {"request": request, "error": str(exc)}, status_code=500
    )


# Logout route
@app.get("/logout", response_class=HTMLResponse)
async def logout(request: Request, response: Response, db: Database = Depends(get_db)):
    await del_user_info_from_cookie(request=request, response=response, db=db)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(key="session_token", value="", expires=0)
    response.set_cookie(key="X-User-Info", value="", expires=0)
    return response


# Get tokens left route
@app.get("/get-tokens-left")
async def get_tokens_left(request: Request, db: Database = Depends(get_db)):
    try:
        user_info = await get_user_info_from_cookie(request)
        user_details = await get_user_details(user_info["email"])
        await reset_tokens_for_user(
            user_details,
            config["token_config"]["tokens_left"],
            config["token_config"]["regen_time"],
        )
        tokens_left = user_details.metadata["tokens_left"]
        return {"tokens_left": tokens_left}
    except Exception as e:
        print(f"Error getting tokens left: {e}")
        return {"tokens_left": 0}


# Mount Chainlit
mount_chainlit(app=app, target="chainlit_app.py", path=CHAINLIT_PATH)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
