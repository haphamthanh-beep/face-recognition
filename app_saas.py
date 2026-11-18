import base64
from datetime import datetime
import json
import os
import sqlite3
import urllib.request
import uuid
import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import httpx
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_FILE = os.path.join(BASE_DIR, "saas_faces.db")
CAPTURES_DIR = os.path.join(BASE_DIR, "captured_images")

# Maximum number of photos kept in local folder (FIFO rotation)
MAX_STORED_IMAGES = 200

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

COSINE_SIMILARITY_THRESHOLD = 0.363


def ensure_models_exist():
    for model_path, url in [
        (YUNET_MODEL_PATH, YUNET_URL),
        (SFACE_MODEL_PATH, SFACE_URL),
    ]:
        if not os.path.exists(model_path):
            print(f"Downloading {os.path.basename(model_path)}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(
                model_path, "wb"
            ) as out_file:
                out_file.write(resp.read())


DEFAULT_BODY_TEMPLATE = """{
  "employeeId": "{{id}}",
  "fullName": "{{name}}",
  "checkInTime": "{{timestamp}}",
  "imagePath": "{{image_path}}",
  "imageUrl": "{{image_url}}",
  "similarity": {{similarity}},
  "device": "Kiosk_01"
}"""


def init_database():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                user_id TEXT NOT NULL,
                full_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tenant_id, user_id)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhook_config (
                tenant_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                method TEXT DEFAULT 'POST',
                headers TEXT DEFAULT '{"Content-Type": "application/json"}',
                body_template TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS capture_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                user_id TEXT,
                full_name TEXT,
                image_filename TEXT NOT NULL,
                similarity REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute(
            """
            INSERT OR IGNORE INTO webhook_config (tenant_id, url, method, headers, body_template)
            VALUES ('default', 'http://localhost:8000/api/v1/mock/attendance-webhook', 'POST', '{"Content-Type": "application/json"}', ?)
            """,
            (DEFAULT_BODY_TEMPLATE,),
        )
        conn.commit()


ensure_models_exist()
init_database()

detector = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)
recognizer = cv2.FaceRecognizerSF.create(model=SFACE_MODEL_PATH, config="")

latest_mock_webhook_event = {
    "status": "WAITING",
    "message": "Chưa có webhook callback nào được gửi đến.",
    "received_at": None,
    "payload": None,
}

app = FastAPI(
    title="Presence Camera Kiosk API",
    description="Captures image on 3s presence, rotates 200 local images max, and previews 10 recent photos.",
    version="1.7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prune_old_images(max_limit: int = MAX_STORED_IMAGES):
    try:
        files = [
            os.path.join(CAPTURES_DIR, f)
            for f in os.listdir(CAPTURES_DIR)
            if f.endswith(".jpg")
        ]
        if len(files) > max_limit:
            files.sort(key=os.path.getmtime)
            num_to_delete = len(files) - max_limit
            for i in range(num_to_delete):
                try:
                    os.remove(files[i])
                except OSError:
                    pass

        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                """
                DELETE FROM capture_events WHERE id NOT IN (
                    SELECT id FROM capture_events ORDER BY id DESC LIMIT ?
                )
                """,
                (max_limit,),
            )
            conn.commit()
    except Exception as e:
        print(f"Error during image pruning: {e}")


def decode_image(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Could not decode."
        )
    return image


def decode_image_and_bytes(raw_input: Union[bytes, str]) -> Tuple[np.ndarray, bytes]:
    if isinstance(raw_input, str):
        cleaned = raw_input.strip()
        if "," in cleaned:
            cleaned = cleaned.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(cleaned)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Invalid base64 encoded image string."
            )
    else:
        image_bytes = raw_input
    return decode_image(image_bytes), image_bytes


def decode_image_from_base64_or_bytes(raw_input: Union[bytes, str]) -> np.ndarray:
    img, _ = decode_image_and_bytes(raw_input)
    return img


def extract_face_embedding(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise HTTPException(
            status_code=400, detail="No face detected in the provided image."
        )
    target_face = faces[0]
    aligned_face = recognizer.alignCrop(image, target_face)
    feature = recognizer.feature(aligned_face)
    return feature


def render_template_string(template_str: str, context: dict) -> str:
    result = template_str
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


async def execute_custom_webhook(
    url_template: str,
    method: str,
    headers_dict: dict,
    body_template: str,
    context: dict,
):
    target_url = render_template_string(url_template, context)
    rendered_body = render_template_string(body_template, context)

    print(f"\n[PRESENCE TRIGGER 3S] Delivering: {method} {target_url}")
    print(f"[PRESENCE TRIGGER 3S] Rendered Body:\n{rendered_body}")

    try:
        is_json = False
        try:
            payload_json = json.loads(rendered_body)
            is_json = True
        except Exception:
            pass

        async with httpx.AsyncClient(timeout=5.0) as client:
            if is_json:
                res = await client.request(
                    method=method,
                    url=target_url,
                    json=payload_json,
                    headers=headers_dict,
                )
            else:
                res = await client.request(
                    method=method,
                    url=target_url,
                    content=rendered_body,
                    headers=headers_dict,
                )
            print(f"[PRESENCE TRIGGER 3S] Result: HTTP {res.status_code}")
    except Exception as e:
        print(f"[PRESENCE TRIGGER 3S] Delivery error: {str(e)}")


# ----------------------------------------------------
# SERVE IMAGES & RECENT CAPTURES
# ----------------------------------------------------


@app.get("/captures/{filename}")
def get_captured_image(filename: str):
    file_path = os.path.join(CAPTURES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Captured image not found.")
    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/v1/captures/recent")
def get_recent_captures(request: Request, limit: int = 10):
    base_url = str(request.base_url).rstrip("/")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            """
            SELECT image_filename, user_id, full_name, similarity, timestamp
            FROM capture_events
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    result = []
    for filename, uid, name, sim, ts in rows:
        result.append(
            {
                "filename": filename,
                "url": f"{base_url}/captures/{filename}",
                "user_id": uid or "UNKNOWN",
                "name": name or "Khách / Chưa đăng ký",
                "similarity": round(float(sim or 0), 2),
                "timestamp": str(ts) if ts else "",
            }
        )

    total_on_disk = len(
        [f for f in os.listdir(CAPTURES_DIR) if f.endswith(".jpg")]
    )

    return {
        "total_saved_on_disk": total_on_disk,
        "max_capacity": MAX_STORED_IMAGES,
        "recent_captures": result,
    }


# ----------------------------------------------------
# WEBHOOK CONFIG APIS
# ----------------------------------------------------


class WebhookConfigModel(BaseModel):
    tenant_id: str = "default"
    url: str
    method: str = "POST"
    headers: str = '{"Content-Type": "application/json"}'
    body_template: str


@app.get("/api/v1/config/webhook")
def get_webhook_config(tenant_id: str = "default"):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT url, method, headers, body_template FROM webhook_config WHERE tenant_id = ?",
            (tenant_id,),
        )
        row = cursor.fetchone()

    if not row:
        return {
            "tenant_id": tenant_id,
            "url": "http://localhost:8000/api/v1/mock/attendance-webhook",
            "method": "POST",
            "headers": '{"Content-Type": "application/json"}',
            "body_template": DEFAULT_BODY_TEMPLATE,
        }

    return {
        "tenant_id": tenant_id,
        "url": row[0],
        "method": row[1],
        "headers": row[2],
        "body_template": row[3],
    }


@app.post("/api/v1/config/webhook")
def save_webhook_config(payload: WebhookConfigModel):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO webhook_config (tenant_id, url, method, headers, body_template, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(tenant_id) DO UPDATE SET
                url = excluded.url,
                method = excluded.method,
                headers = excluded.headers,
                body_template = excluded.body_template,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                payload.tenant_id,
                payload.url,
                payload.method.upper(),
                payload.headers,
                payload.body_template,
            ),
        )
        conn.commit()

    return {
        "success": True,
        "message": "Cấu hình Webhook Backend đã được lưu thành công.",
    }


# ----------------------------------------------------
# PRESENCE DETECTION & TRIGGER
# ----------------------------------------------------


class PresenceDetectJsonRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64 encoded camera frame image string",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJ..."],
    )
    dwell_completed: bool = Field(
        False, description="Set to True when person has stood for 3 seconds"
    )
    tenant_id: str = Field("default", description="Tenant or organization ID")


async def process_presence_frame(
    image: np.ndarray,
    contents: bytes,
    tenant_id: str,
    dwell_completed: bool,
    base_url: str,
    background_tasks: BackgroundTasks,
) -> dict:
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return {
            "has_person": False,
            "is_registered": False,
            "status": "NO_PERSON",
            "message": "Không có người trước camera.",
        }

    target_face = faces[0]
    aligned_face = recognizer.alignCrop(image, target_face)
    input_feature = recognizer.feature(aligned_face)

    best_user_id = None
    best_name = None
    best_score = 0.0

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT user_id, full_name, embedding FROM face_identities WHERE tenant_id = ?",
            (tenant_id,),
        )
        user_rows = cursor.fetchall()

    for uid, name, emb_bytes in user_rows:
        stored_feature = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, 128)
        score = recognizer.match(
            input_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE
        )
        if score > best_score:
            best_score = score
            if score >= COSINE_SIMILARITY_THRESHOLD:
                best_user_id = uid
                best_name = name

    # Reject if person is not registered or similarity is below threshold
    if not best_user_id or best_score < COSINE_SIMILARITY_THRESHOLD:
        return {
            "has_person": True,
            "is_registered": False,
            "status": "UNREGISTERED",
            "similarity": round(float(best_score), 4),
            "message": "Khuôn mặt chưa đăng ký (không đếm giờ, không chụp).",
        }

    # Registered person is in front of the camera, but 3 seconds not completed yet
    if not dwell_completed:
        return {
            "has_person": True,
            "is_registered": True,
            "status": "COUNTING",
            "user_id": best_user_id,
            "name": best_name,
            "similarity": round(float(best_score), 4),
            "message": f"Đã nhận diện: {best_name} ({best_user_id}), đang có mặt...",
        }

    # Registered person has stood for 3 full seconds: save image and fire webhook
    now = datetime.now()
    now_iso = now.isoformat()
    now_time = now.strftime("%H:%M:%S")
    now_date = now.strftime("%Y-%m-%d")

    filename = f"capture_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    saved_file_path = os.path.join(CAPTURES_DIR, filename)
    with open(saved_file_path, "wb") as f:
        f.write(contents)

    prune_old_images(MAX_STORED_IMAGES)

    image_url = f"{base_url}/captures/{filename}"
    image_base64 = base64.b64encode(contents).decode("utf-8")

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT url, method, headers, body_template FROM webhook_config WHERE tenant_id = ?",
            (tenant_id,),
        )
        cfg_row = cursor.fetchone()

        conn.execute(
            "INSERT INTO capture_events (tenant_id, user_id, full_name, image_filename, similarity) VALUES (?, ?, ?, ?, ?)",
            (tenant_id, best_user_id, best_name, filename, float(best_score)),
        )
        conn.commit()

    if cfg_row:
        cfg_url, cfg_method, cfg_headers_raw, cfg_body_template = cfg_row
    else:
        cfg_url = "http://localhost:8000/api/v1/mock/attendance-webhook"
        cfg_method = "POST"
        cfg_headers_raw = '{"Content-Type": "application/json"}'
        cfg_body_template = DEFAULT_BODY_TEMPLATE

    try:
        headers_dict = json.loads(cfg_headers_raw)
    except Exception:
        headers_dict = {"Content-Type": "application/json"}

    context = {
        "id": best_user_id,
        "user_id": best_user_id,
        "name": best_name,
        "full_name": best_name,
        "image_path": saved_file_path,
        "image_url": image_url,
        "image_base64": image_base64,
        "filename": filename,
        "timestamp": now_iso,
        "time": now_time,
        "date": now_date,
        "similarity": f"{best_score:.4f}",
        "tenant_id": tenant_id,
    }

    background_tasks.add_task(
        execute_custom_webhook,
        cfg_url,
        cfg_method,
        headers_dict,
        cfg_body_template,
        context,
    )

    return {
        "has_person": True,
        "is_registered": True,
        "status": "TRIGGERED",
        "message": f"Người đã đăng ký ({best_name}) có mặt đủ 3s! Đã lưu ảnh {filename} và gửi Webhook.",
        "image_saved": filename,
        "image_path": saved_file_path,
        "image_url": image_url,
        "identified_user_id": best_user_id,
        "identified_name": best_name,
        "similarity": round(float(best_score), 4),
        "webhook_target": cfg_url,
    }


@app.post(
    "/api/v1/presence/detect-and-trigger",
    summary="Detect presence and trigger (Multipart File)",
    tags=["Presence & Kiosk"],
)
async def detect_presence_and_trigger(
    background_tasks: BackgroundTasks,
    request: Request,
    tenant_id: str = Form("default"),
    dwell_completed: bool = Form(
        False, description="Set to True when registered person has stood for 3 seconds"
    ),
    file: UploadFile = File(..., description="Camera snapshot frame"),
):
    contents = await file.read()
    image = decode_image(contents)
    base_url = str(request.base_url).rstrip("/")
    return await process_presence_frame(
        image=image,
        contents=contents,
        tenant_id=tenant_id,
        dwell_completed=dwell_completed,
        base_url=base_url,
        background_tasks=background_tasks,
    )


@app.post(
    "/api/v1/presence/detect-and-trigger-json",
    summary="Detect presence and trigger (JSON Base64)",
    tags=["Presence & Kiosk"],
)
async def detect_presence_and_trigger_json(
    payload: PresenceDetectJsonRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    image, contents = decode_image_and_bytes(payload.image_base64)
    base_url = str(request.base_url).rstrip("/")
    return await process_presence_frame(
        image=image,
        contents=contents,
        tenant_id=payload.tenant_id,
        dwell_completed=payload.dwell_completed,
        base_url=base_url,
        background_tasks=background_tasks,
    )


# ----------------------------------------------------
# MOCK WEBHOOK RECEIVER
# ----------------------------------------------------


@app.post("/api/v1/mock/attendance-webhook")
@app.put("/api/v1/mock/attendance-webhook")
async def mock_backend_webhook_receiver(request: Request):
    global latest_mock_webhook_event
    now_str = datetime.now().strftime("%H:%M:%S")

    try:
        payload = await request.json()
    except Exception:
        body_bytes = await request.body()
        payload = body_bytes.decode("utf-8")

    latest_mock_webhook_event = {
        "status": "RECEIVED_SUCCESS",
        "received_at": now_str,
        "headers": dict(request.headers),
        "payload": payload,
    }
    print(f"\n✅ [MOCK BACKEND] Received Callback at {now_str}:\n{payload}")
    return {"status": "ACK", "message": "Backend received photo capture & webhook."}


@app.get("/api/v1/mock/latest-callback")
def get_latest_mock_callback():
    return latest_mock_webhook_event


# ----------------------------------------------------
# REGISTRATION, CRUD & VERIFICATION APIS
# ----------------------------------------------------


class RegisterFaceRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="Unique employee / user ID",
        examples=["NV-001"],
    )
    full_name: str = Field(
        ...,
        description="Employee full name",
        examples=["Nguyen Van A"],
    )
    image_base64: str = Field(
        ...,
        description="Base64 encoded face image string (JPEG/PNG, with or without data:image/... prefix)",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJ..."],
    )
    tenant_id: str = Field(
        "default",
        description="Tenant or organization ID",
        examples=["default"],
    )


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = Field(
        None,
        description="Updated full name",
        examples=["Nguyen Van A (Cap Nhat)"],
    )
    image_base64: Optional[str] = Field(
        None,
        description="Optional new base64 face image if replacing face template",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJ..."],
    )
    tenant_id: str = Field(
        "default",
        description="Tenant or organization ID",
        examples=["default"],
    )


class VerifyFaceRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="Employee / User ID to verify against",
        examples=["1111"],
    )
    image_base64: str = Field(
        ...,
        description="Base64 encoded face image to verify",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJ..."],
    )
    tenant_id: str = Field(
        "default",
        description="Tenant or organization ID",
        examples=["default"],
    )


class IdentifyFaceRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64 encoded face image to identify across all registered users",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJ..."],
    )
    tenant_id: str = Field(
        "default",
        description="Tenant or organization ID",
        examples=["default"],
    )


# Backward compatibility aliases
RegisterFaceJsonRequest = RegisterFaceRequest
UpdateUserJsonRequest = UpdateUserRequest
VerifyFaceJsonRequest = VerifyFaceRequest
IdentifyFaceJsonRequest = IdentifyFaceRequest


# 1. CREATE / REGISTER FACE (JSON Base64) - PRIMARY
@app.post(
    "/api/v1/faces/register",
    summary="Register a new face (JSON Base64)",
    tags=["Face Identity Management"],
)
@app.post(
    "/api/v1/faces/register-json",
    summary="Register a new face (JSON Base64 - Alias)",
    tags=["Face Identity Management"],
    include_in_schema=False,
)
def register_face(payload: RegisterFaceRequest):
    image = decode_image_from_base64_or_bytes(payload.image_base64)
    feature = extract_face_embedding(image)
    embedding_bytes = feature.astype(np.float32).tobytes()

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO face_identities (tenant_id, user_id, full_name, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                full_name = excluded.full_name,
                embedding = excluded.embedding,
                created_at = CURRENT_TIMESTAMP
            """,
            (payload.tenant_id, payload.user_id, payload.full_name, embedding_bytes),
        )
        conn.commit()

    return {
        "success": True,
        "user_id": payload.user_id,
        "full_name": payload.full_name,
        "tenant_id": payload.tenant_id,
        "message": "Face registered successfully via Base64 JSON payload.",
    }


# 2. CREATE / REGISTER FACE (Multipart file - Alternative)
@app.post(
    "/api/v1/faces/register-file",
    summary="Register a new face (File Upload)",
    tags=["Face Identity Management"],
)
async def register_face_file(
    user_id: str = Form(..., description="Unique employee / user ID"),
    full_name: str = Form(..., description="Employee full name"),
    tenant_id: str = Form("default", description="Tenant or organization ID"),
    file: UploadFile = File(..., description="Face image file (JPEG/PNG)"),
):
    contents = await file.read()
    image = decode_image(contents)
    feature = extract_face_embedding(image)
    embedding_bytes = feature.astype(np.float32).tobytes()

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO face_identities (tenant_id, user_id, full_name, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                full_name = excluded.full_name,
                embedding = excluded.embedding,
                created_at = CURRENT_TIMESTAMP
            """,
            (tenant_id, user_id, full_name, embedding_bytes),
        )
        conn.commit()

    return {
        "success": True,
        "user_id": user_id,
        "full_name": full_name,
        "tenant_id": tenant_id,
        "message": "Face registered successfully via file upload.",
    }


# 3. LIST USERS
@app.get(
    "/api/v1/faces/users",
    summary="List all registered users",
    tags=["Face Identity Management"],
)
def list_users(
    tenant_id: str = "default",
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    with sqlite3.connect(DB_FILE) as conn:
        if search:
            search_pattern = f"%{search}%"
            cursor = conn.execute(
                """
                SELECT user_id, full_name, created_at FROM face_identities
                WHERE tenant_id = ? AND (user_id LIKE ? OR full_name LIKE ?)
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (tenant_id, search_pattern, search_pattern, limit, offset),
            )
            rows = cursor.fetchall()
            count_cursor = conn.execute(
                """
                SELECT COUNT(*) FROM face_identities
                WHERE tenant_id = ? AND (user_id LIKE ? OR full_name LIKE ?)
                """,
                (tenant_id, search_pattern, search_pattern),
            )
            total = count_cursor.fetchone()[0]
        else:
            cursor = conn.execute(
                """
                SELECT user_id, full_name, created_at FROM face_identities
                WHERE tenant_id = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (tenant_id, limit, offset),
            )
            rows = cursor.fetchall()
            count_cursor = conn.execute(
                "SELECT COUNT(*) FROM face_identities WHERE tenant_id = ?",
                (tenant_id,),
            )
            total = count_cursor.fetchone()[0]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "users": [
            {"user_id": r[0], "full_name": r[1], "created_at": r[2]} for r in rows
        ],
    }


# 4. GET SINGLE USER
@app.get(
    "/api/v1/faces/users/{user_id}",
    summary="Get user face identity details",
    tags=["Face Identity Management"],
)
def get_user(user_id: str, tenant_id: str = "default"):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT user_id, full_name, created_at FROM face_identities WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found in tenant '{tenant_id}'.",
        )

    return {
        "user_id": row[0],
        "full_name": row[1],
        "tenant_id": tenant_id,
        "created_at": row[2],
        "has_face_template": True,
    }


# 5. UPDATE USER (JSON Payload)
@app.put(
    "/api/v1/faces/users/{user_id}",
    summary="Update user information or face template (JSON)",
    tags=["Face Identity Management"],
)
def update_user(user_id: str, payload: UpdateUserJsonRequest):
    tenant_id = payload.tenant_id

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT full_name FROM face_identities WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"User '{user_id}' not found in tenant '{tenant_id}'.",
            )

        updated_name = payload.full_name if payload.full_name else existing[0]

        if payload.image_base64:
            image = decode_image_from_base64_or_bytes(payload.image_base64)
            feature = extract_face_embedding(image)
            embedding_bytes = feature.astype(np.float32).tobytes()

            conn.execute(
                """
                UPDATE face_identities
                SET full_name = ?, embedding = ?, created_at = CURRENT_TIMESTAMP
                WHERE tenant_id = ? AND user_id = ?
                """,
                (updated_name, embedding_bytes, tenant_id, user_id),
            )
        else:
            conn.execute(
                """
                UPDATE face_identities
                SET full_name = ?
                WHERE tenant_id = ? AND user_id = ?
                """,
                (updated_name, tenant_id, user_id),
            )
        conn.commit()

    return {
        "success": True,
        "user_id": user_id,
        "full_name": updated_name,
        "face_updated": bool(payload.image_base64),
        "message": f"User '{user_id}' updated successfully.",
    }


# 6. UPDATE USER (Multipart Form)
@app.put(
    "/api/v1/faces/users/{user_id}/form",
    summary="Update user information or face template (Multipart Form)",
    tags=["Face Identity Management"],
)
async def update_user_form(
    user_id: str,
    full_name: Optional[str] = Form(None),
    tenant_id: str = Form("default"),
    file: Optional[UploadFile] = File(None),
):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT full_name FROM face_identities WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"User '{user_id}' not found in tenant '{tenant_id}'.",
            )

        updated_name = full_name if full_name else existing[0]

        if file:
            contents = await file.read()
            image = decode_image(contents)
            feature = extract_face_embedding(image)
            embedding_bytes = feature.astype(np.float32).tobytes()

            conn.execute(
                """
                UPDATE face_identities
                SET full_name = ?, embedding = ?, created_at = CURRENT_TIMESTAMP
                WHERE tenant_id = ? AND user_id = ?
                """,
                (updated_name, embedding_bytes, tenant_id, user_id),
            )
        else:
            conn.execute(
                """
                UPDATE face_identities
                SET full_name = ?
                WHERE tenant_id = ? AND user_id = ?
                """,
                (updated_name, tenant_id, user_id),
            )
        conn.commit()

    return {
        "success": True,
        "user_id": user_id,
        "full_name": updated_name,
        "face_updated": bool(file),
        "message": f"User '{user_id}' updated successfully.",
    }


# 7. DELETE USER
@app.delete(
    "/api/v1/faces/users/{user_id}",
    summary="Delete user face identity",
    tags=["Face Identity Management"],
)
def delete_user(user_id: str, tenant_id: str = "default"):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "DELETE FROM face_identities WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=f"User '{user_id}' not found in tenant '{tenant_id}'.",
            )

    return {
        "success": True,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "message": f"User '{user_id}' deleted successfully.",
    }


# 8. VERIFY 1:1 FACE (JSON Base64)
@app.post(
    "/api/v1/faces/verify",
    summary="1:1 Face Verification (JSON Base64)",
    tags=["Face Verification & Identification"],
)
def verify_face_json(payload: VerifyFaceJsonRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT full_name, embedding FROM face_identities WHERE tenant_id = ? AND user_id = ?",
            (payload.tenant_id, payload.user_id),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"User '{payload.user_id}' not found in tenant '{payload.tenant_id}'.",
        )

    full_name, emb_bytes = row
    stored_feature = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, 128)

    image = decode_image_from_base64_or_bytes(payload.image_base64)
    input_feature = extract_face_embedding(image)

    score = recognizer.match(
        input_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE
    )
    is_match = bool(score >= COSINE_SIMILARITY_THRESHOLD)

    return {
        "is_match": is_match,
        "user_id": payload.user_id,
        "full_name": full_name,
        "similarity": round(float(score), 4),
        "threshold": COSINE_SIMILARITY_THRESHOLD,
        "status": "MATCH" if is_match else "MISMATCH",
        "message": (
            f"Face matches {full_name} ({payload.user_id})."
            if is_match
            else "Face does not match the registered user."
        ),
    }


# 9. VERIFY 1:1 FACE (Multipart File)
@app.post(
    "/api/v1/faces/verify-file",
    summary="1:1 Face Verification (Multipart File)",
    tags=["Face Verification & Identification"],
)
async def verify_face_file(
    user_id: str = Form(..., description="Employee / User ID to verify against"),
    tenant_id: str = Form("default", description="Tenant ID"),
    file: UploadFile = File(..., description="Face image file to verify"),
):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT full_name, embedding FROM face_identities WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found in tenant '{tenant_id}'.",
        )

    full_name, emb_bytes = row
    stored_feature = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, 128)

    contents = await file.read()
    image = decode_image(contents)
    input_feature = extract_face_embedding(image)

    score = recognizer.match(
        input_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE
    )
    is_match = bool(score >= COSINE_SIMILARITY_THRESHOLD)

    return {
        "is_match": is_match,
        "user_id": user_id,
        "full_name": full_name,
        "similarity": round(float(score), 4),
        "threshold": COSINE_SIMILARITY_THRESHOLD,
        "status": "MATCH" if is_match else "MISMATCH",
        "message": (
            f"Face matches {full_name} ({user_id})."
            if is_match
            else "Face does not match the registered user."
        ),
    }


# 10. IDENTIFY 1:N FACE (JSON Base64)
@app.post(
    "/api/v1/faces/identify",
    summary="1:N Face Identification (JSON Base64)",
    tags=["Face Verification & Identification"],
)
def identify_face_json(payload: IdentifyFaceJsonRequest):
    image = decode_image_from_base64_or_bytes(payload.image_base64)
    input_feature = extract_face_embedding(image)

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT user_id, full_name, embedding FROM face_identities WHERE tenant_id = ?",
            (payload.tenant_id,),
        )
        user_rows = cursor.fetchall()

    if not user_rows:
        return {
            "is_identified": False,
            "user_id": None,
            "full_name": None,
            "similarity": 0.0,
            "threshold": COSINE_SIMILARITY_THRESHOLD,
            "status": "NO_REGISTERED_USERS",
            "message": "No registered users in this tenant.",
        }

    best_user_id = None
    best_name = None
    best_score = 0.0

    for uid, name, emb_bytes in user_rows:
        stored_feature = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, 128)
        score = recognizer.match(
            input_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE
        )
        if score > best_score:
            best_score = score
            if score >= COSINE_SIMILARITY_THRESHOLD:
                best_user_id = uid
                best_name = name

    is_identified = bool(best_user_id and best_score >= COSINE_SIMILARITY_THRESHOLD)

    return {
        "is_identified": is_identified,
        "user_id": best_user_id,
        "full_name": best_name,
        "similarity": round(float(best_score), 4),
        "threshold": COSINE_SIMILARITY_THRESHOLD,
        "status": "IDENTIFIED" if is_identified else "UNKNOWN",
        "message": (
            f"Identified as {best_name} ({best_user_id})"
            if is_identified
            else "Face does not match any registered user."
        ),
    }


# 11. IDENTIFY 1:N FACE (Multipart File)
@app.post(
    "/api/v1/faces/identify-file",
    summary="1:N Face Identification (Multipart File)",
    tags=["Face Verification & Identification"],
)
async def identify_face_file(
    tenant_id: str = Form("default", description="Tenant ID"),
    file: UploadFile = File(..., description="Face image file to identify"),
):
    contents = await file.read()
    image = decode_image(contents)
    input_feature = extract_face_embedding(image)

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT user_id, full_name, embedding FROM face_identities WHERE tenant_id = ?",
            (tenant_id,),
        )
        user_rows = cursor.fetchall()

    if not user_rows:
        return {
            "is_identified": False,
            "user_id": None,
            "full_name": None,
            "similarity": 0.0,
            "threshold": COSINE_SIMILARITY_THRESHOLD,
            "status": "NO_REGISTERED_USERS",
            "message": "No registered users in this tenant.",
        }

    best_user_id = None
    best_name = None
    best_score = 0.0

    for uid, name, emb_bytes in user_rows:
        stored_feature = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, 128)
        score = recognizer.match(
            input_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE
        )
        if score > best_score:
            best_score = score
            if score >= COSINE_SIMILARITY_THRESHOLD:
                best_user_id = uid
                best_name = name

    is_identified = bool(best_user_id and best_score >= COSINE_SIMILARITY_THRESHOLD)

    return {
        "is_identified": is_identified,
        "user_id": best_user_id,
        "full_name": best_name,
        "similarity": round(float(best_score), 4),
        "threshold": COSINE_SIMILARITY_THRESHOLD,
        "status": "IDENTIFIED" if is_identified else "UNKNOWN",
        "message": (
            f"Identified as {best_name} ({best_user_id})"
            if is_identified
            else "Face does not match any registered user."
        ),
    }


# ----------------------------------------------------
# SYSTEM & ROOT INFO (PURE BACKEND SERVICE)
# ----------------------------------------------------


@app.get(
    "/",
    summary="Root Service Info & Health",
    tags=["System"],
)
def root_service_info():
    return {
        "service": "Face Recognition & Identity Microservice",
        "status": "UP",
        "version": "2.0.0",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "description": "Pure REST API Service for Face Registration, Update, Delete, 1:1 Verification, and 1:N Identification using Base64 JSON payloads.",
        "endpoints": {
            "register": "POST /api/v1/faces/register (JSON Base64)",
            "verify": "POST /api/v1/faces/verify (JSON Base64)",
            "identify": "POST /api/v1/faces/identify (JSON Base64)",
            "users": "GET /api/v1/faces/users",
            "get_user": "GET /api/v1/faces/users/{user_id}",
            "update_user": "PUT /api/v1/faces/users/{user_id}",
            "delete_user": "DELETE /api/v1/faces/users/{user_id}",
            "presence_trigger": "POST /api/v1/presence/detect-and-trigger-json",
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 65)
    print("🚀 Face Recognition Microservice API (Pure Service)")
    print(" - API Server Base URL       : http://localhost:8000")
    print(" - Interactive Swagger UI    : http://localhost:8000/docs")
    print(" - OpenAPI Spec (JSON)       : http://localhost:8000/openapi.json")
    print(" - Local Captures Directory  : " + CAPTURES_DIR)
    print("=" * 65 + "\n")
    uvicorn.run("app_saas:app", host="0.0.0.0", port=8000, reload=True)
