#!/bin/bash
cd "$(dirname "$0")"

echo "=================================================="
echo "    HE THONG NHAN DIEN & DIEM DANH KHUON MAT    "
echo "=================================================="
echo "  1. Chay SAAS Microservice (Web UI + Swagger API)"
echo "  2. Phase 1: Dang ky mat bang Desktop Camera"
echo "  3. Phase 2: Diem danh bang Desktop Camera"
echo "  4. Thoat"
echo "=================================================="
read -p "Chon chuc nang (1, 2, 3, hoac 4): " choice

case $choice in
  1)
    echo "Dang khoi chay SaaS Microservice tai http://localhost:8000 ..."
    ./venv/bin/python3 app_saas.py
    ;;
  2)
    echo "Dang khoi chay Phase 1: Dang ky..."
    ./venv/bin/python3 register_face.py
    ;;
  3)
    echo "Dang khoi chay Phase 2: Diem danh..."
    ./venv/bin/python3 attendance_checkin.py
    ;;
  4)
    echo "Tam biet!"
    exit 0
    ;;
  *)
    echo "Lua chon khong hop le!"
    exit 1
    ;;
esac
