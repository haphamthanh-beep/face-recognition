# Face Recognition & Verification Microservice

Hệ thống nhận diện và xác thực khuôn mặt (Face Recognition & 1:1 Verification & 1:N Identification) kiến trúc Microservice, sử dụng hoàn toàn các công nghệ mã nguồn mở miễn phí: **FastAPI**, **OpenCV YuNet** (Face Detection), **OpenCV SFace** (Face Recognition), **SQLite** và **TailwindCSS**.

Dự án được phân tách thành 2 thành phần độc lập:
1. **Backend Service (`app_saas.py`)**: Cung cấp Pure REST API (nhận và trả JSON Base64).
2. **Webcam Test Client (`client_server.py` / `client_test/`)**: Giao diện kiểm thử độc lập dùng Webcam máy tính để Thêm, Sửa, Xoá, Verify (1:1) và Identify (1:N).

---

## 🚀 Tính Năng Chính

- **Pure REST API (JSON Base64)**: Hỗ trợ tích hợp đa nền tảng (Java/Spring Boot, Node.js, Go, Python...).
- **Không tốn dung lượng lưu trữ ảnh**: Ảnh đăng ký được trích xuất thành vector toán học 128 chiều (`BLOB`, 512 bytes) lưu trong SQLite. Không lưu ảnh thô, đảm bảo tốc độ truy vấn mili-giây và an toàn dữ liệu.
- **Xác thực 1:1 (Verification)**: So khớp khuôn mặt trước camera với một nhân viên cụ thể xem có đúng người không.
- **Nhận diện 1:N (Identification)**: Quét khuôn mặt trước camera và tự động tìm ra danh tính nhân viên trong toàn bộ cơ sở dữ liệu (hoặc cảnh báo người lạ).
- **Kiosk Auto-Trigger & Webhook**: Chế độ tự động đếm đủ 3 giây khi có nhân viên hợp lệ đứng trước camera, tự động chụp ảnh, lưu trữ xoay vòng tối đa 200 ảnh và bắn Webhook callback về hệ thống backend doanh nghiệp.
- **Webcam Test Client Trực Quan**: Giao diện web chạy tại `http://localhost:8088` hỗ trợ camera trực tiếp, tự động bắt frame Base64, hiển thị kết quả và log gói tin API theo thời gian thực.

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ**: Python 3.9+
- **Web Framework**: FastAPI, Uvicorn, Pydantic v2
- **AI Models (OpenCV Zoo - Tự động tải về khi chạy lần đầu)**:
  - *Face Detection*: YuNet (`models/face_detection_yunet.onnx`)
  - *Face Recognition*: SFace (`models/face_recognition_sface.onnx`)
- **Database**: SQLite (`saas_faces.db`)
- **Test Client UI**: HTML5 Canvas, WebRTC Camera API, Tailwind CSS, FontAwesome

---

## 📦 Cấu Trúc Thư Mục

```text
face-recognition/
├── app_saas.py                     # Backend API Microservice (Port 8000)
├── client_server.py                # Server phục vụ Web Test Client (Port 8088)
├── client_test/
│   └── index.html                  # Giao diện Web Test Client Webcam
├── models/                         # Thư mục lưu trữ model ONNX (YuNet & SFace)
│   └── .gitkeep
├── sample_face_base64.txt          # Chuỗi Base64 ảnh mẫu hợp lệ để test nhanh
├── Face_Recognition_API.postman_collection.json  # Bộ Postman Collection v2.1
├── openapi.json                    # File OpenAPI 3.1 Specification
├── requirements.txt                # Thư viện Python phụ thuộc
├── run.sh                          # Script khởi chạy Backend Service
├── run_client.sh                   # Script khởi chạy Test Client
└── README.md
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Chuẩn bị môi trường Python

```bash
# Clone dự án
git clone https://github.com/haphamthanh-beep/face-recognition.git
cd face-recognition

# Tạo môi trường ảo (virtual environment)
python3 -m venv venv
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

### 2. Khởi chạy Backend API Service

```bash
# Chạy Backend Service tại cổng 8000
python3 app_saas.py
# hoặc dùng script
./run.sh
```

- **API Base URL**: `http://localhost:8000`
- **Tài liệu tương tác Swagger UI**: `http://localhost:8000/docs`
- **Tài liệu ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

*(Ở lần chạy đầu tiên, hệ thống sẽ tự động tải 2 file mô hình AI YuNet và SFace vào thư mục `models/`).*

---

### 3. Khởi chạy Webcam Test Client

Mở một tab terminal mới:

```bash
source venv/bin/activate
python3 client_server.py 8088
# hoặc dùng script
./run_client.sh
```

Mở trình duyệt truy cập: **`http://localhost:8088`**
- Bật webcam và cấp quyền camera cho trình duyệt.
- Bạn có thể thực hiện Thêm nhân viên mới, Verify 1:1, Identify 1:N, Sửa thông tin và Xóa nhân viên trực tiếp qua Webcam.

---

## 📡 Danh Sách API (REST Endpoints)

Tất cả các API nhận ảnh đều hỗ trợ chuỗi Base64 thuần (`/9j/4AAQ...`) hoặc Data URI (`data:image/jpeg;base64,...`).

| Chức năng | Method | Endpoint | Định dạng |
|---|---|---|---|
| **Đăng ký (Add)** | `POST` | `/api/v1/faces/register` | JSON Base64 |
| **So khớp 1:1 (Verify)** | `POST` | `/api/v1/faces/verify` | JSON Base64 |
| **Nhận diện 1:N (Identify)**| `POST` | `/api/v1/faces/identify` | JSON Base64 |
| **Danh sách nhân viên** | `GET` | `/api/v1/faces/users` | Query params (`search`, `limit`, `offset`) |
| **Chi tiết nhân viên** | `GET` | `/api/v1/faces/users/{user_id}` | Path param |
| **Cập nhật nhân viên** | `PUT` | `/api/v1/faces/users/{user_id}` | JSON Base64 (Đổi tên / ảnh mới) |
| **Xóa nhân viên** | `DELETE` | `/api/v1/faces/users/{user_id}` | Path param |

---

### Ví Dụ Gói Tin API

#### 1. Đăng ký nhân viên (`POST /api/v1/faces/register`)

**Request**:
```json
{
  "user_id": "NV-001",
  "full_name": "Nguyễn Văn A",
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQ...",
  "tenant_id": "default"
}
```

**Response `200 OK`**:
```json
{
  "success": true,
  "user_id": "NV-001",
  "full_name": "Nguyễn Văn A",
  "tenant_id": "default",
  "message": "Face registered successfully via Base64 JSON payload."
}
```

---

#### 2. Verify 1:1 (`POST /api/v1/faces/verify`)

**Request**:
```json
{
  "user_id": "NV-001",
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQ...",
  "tenant_id": "default"
}
```

**Response khi trùng khớp**:
```json
{
  "is_match": true,
  "user_id": "NV-001",
  "full_name": "Nguyễn Văn A",
  "similarity": 0.8247,
  "threshold": 0.363,
  "status": "MATCH",
  "message": "Face matches Nguyễn Văn A (NV-001)."
}
```

---

#### 3. Identify 1:N (`POST /api/v1/faces/identify`)

**Request**:
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQ...",
  "tenant_id": "default"
}
```

**Response khi tìm thấy**:
```json
{
  "is_identified": true,
  "user_id": "NV-001",
  "full_name": "Nguyễn Văn A",
  "similarity": 0.8247,
  "threshold": 0.363,
  "status": "IDENTIFIED",
  "message": "Identified as Nguyễn Văn A (NV-001)"
}
```

---

## 🧪 Kiểm Thử Bằng Postman

Trong thư mục dự án có sẵn file **[`Face_Recognition_API.postman_collection.json`](Face_Recognition_API.postman_collection.json)**:
1. Mở Postman -> Chọn **Import** -> Chọn file này.
2. Collection đã thiết lập sẵn các biến môi trường:
   - `{{baseUrl}}`: `http://localhost:8000`
   - `{{tenantId}}`: `default`
   - `{{sampleBase64}}`: Đã gán sẵn chuỗi Base64 ảnh mẫu hợp lệ từ file `sample_face_base64.txt`.
3. Bấm **Send** để kiểm thử ngay lập tức từng endpoint.

---

## ☕ Tích Hợp Vào Backend Doanh Nghiệp (Java / Spring Boot)

```java
// Đọc ảnh từ file hoặc camera thành Base64
byte[] fileContent = Files.readAllBytes(Paths.get("/path/to/face.jpg"));
String base64Image = Base64.getEncoder().encodeToString(fileContent);

// Chuẩn bị payload JSON
Map<String, Object> payload = new HashMap<>();
payload.put("user_id", "NV-001");
payload.put("image_base64", base64Image);
payload.put("tenant_id", "default");

HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.APPLICATION_JSON);

HttpEntity<Map<String, Object>> request = new HttpEntity<>(payload, headers);
ResponseEntity<Map> response = restTemplate.postForEntity(
    "http://localhost:8000/api/v1/faces/verify", 
    request, 
    Map.class
);

Boolean isMatch = (Boolean) response.getBody().get("is_match");
Double similarity = (Double) response.getBody().get("similarity");
System.out.println("Kết quả so khớp: " + isMatch + ", Similarity: " + similarity);
```

---

## 📄 License
Dự án được phân phối dưới giấy phép mã nguồn mở MIT.

