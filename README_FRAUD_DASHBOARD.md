# Real-time Fraud Detection Dashboard

Dashboard phát hiện gian lận trực tuyến cho hệ thống ngân hàng, được phát triển theo yêu cầu từ Report 16.11.pdf.

## Tính năng chính

### RTFD-2.1: Alert Generation (Tạo cảnh báo)
- Phát hiện gian lận tự động sử dụng Machine Learning model đã train
- Tính toán Fraud Score (0-1) cho mỗi giao dịch
- Phân loại rủi ro: High (≥70%), Medium (40-70%), Low (<40%)
- Gửi email tự động cho các trường hợp High Risk

### RTFD-2.2: Alert Storage (Lưu trữ cảnh báo)
- Database SQLite lưu trữ tất cả alerts
- Lưu trữ: transaction_id, fraud_score, amount, risk_level, review_status
- Lưu trữ 28 features (V1-V28) từ PCA transformation
- Timestamps và trạng thái email notification

### RTFD-2.3: Dashboard Visualization (Trực quan hóa)
- **KPI Cards**: Tổng giao dịch, tỷ lệ gian lận, alerts theo mức độ rủi ro
- **Fraud Trend Chart**: Biểu đồ xu hướng gian lận 7 ngày qua
- **Risk Distribution Chart**: Phân bổ alerts theo mức độ rủi ro
- **Real-time Statistics**: Cập nhật tự động mỗi 30 giây

### RTFD-2.4: Alert Filtering (Lọc cảnh báo)
- Filter theo: Risk Level, Review Status, Amount Range, Date Range
- Pagination hỗ trợ hiển thị nhiều alerts
- Cập nhật trạng thái review trực tiếp trên bảng

## Công nghệ sử dụng

- **Backend**: Flask (Python) - theo khuyến nghị Report page 59
- **Database**: SQLite với SQLAlchemy ORM
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Charts**: Chart.js cho visualization
- **ML Model**: Random Forest đã train (fraud_detection_model.pkl)

## Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
python app.py
```

Server sẽ chạy tại: `http://localhost:5000`

### 3. Đăng nhập

- **Username**: admin
- **Password**: admin123

## Hướng dẫn sử dụng

### Kiểm tra giao dịch gian lận

1. Vào tab **Transaction Fraud Check**
2. Nhập Transaction ID và Amount
3. Click **Generate Random Test Data** để tạo dữ liệu test (V1-V28)
4. Click **Check for Fraud**
5. Xem kết quả: Fraud Score, Risk Level, Recommendation

### Xem thống kê

- **KPI Cards**: Hiển thị tổng quan 24h gần nhất
- **Fraud Trend**: Xem xu hướng 7 ngày
- **Alert Distribution**: Phân bổ theo mức độ rủi ro

### Quản lý Alerts

1. Vào phần **Alert Management**
2. Sử dụng filters để lọc alerts:
   - Risk Level: High/Medium/Low
   - Review Status: Pending/Reviewed/Resolved
   - Amount Range: Min/Max
3. Cập nhật trạng thái review trực tiếp từ dropdown
4. Click **View** để xem chi tiết alert

## Email Notifications

Để kích hoạt email notifications cho High Risk alerts:

```bash
# Windows
set EMAIL_SENDER=your-email@gmail.com
set EMAIL_RECEIVER=security@bank.com
set EMAIL_PASSWORD=your-app-password

# Linux/Mac
export EMAIL_SENDER=your-email@gmail.com
export EMAIL_RECEIVER=security@bank.com
export EMAIL_PASSWORD=your-app-password
```

**Lưu ý**: Sử dụng App Password cho Gmail, không dùng password thường.

## Cấu trúc API

### GET /api/statistics
Lấy thống kê real-time:
- Total transactions (24h)
- Fraud rate
- High/Medium/Low risk alerts
- Trend data (7 days)

### GET /api/alerts
Lấy danh sách alerts với filters:
- Query params: `risk_level`, `status`, `min_amount`, `max_amount`, `page`, `per_page`

### POST /api/predict
Dự đoán gian lận cho giao dịch mới:
```json
{
  "transaction_id": "TXN-123456",
  "Amount": 1500.50,
  "V1": 0.1234,
  "V2": -0.5678,
  ...
  "V28": 0.9012
}
```

### PUT /api/alerts/:id/status
Cập nhật trạng thái review:
```json
{
  "status": "Reviewed"
}
```

## Database Schema

```sql
CREATE TABLE fraud_alert (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    transaction_id VARCHAR(50) UNIQUE,
    fraud_score FLOAT,
    amount FLOAT,
    risk_level VARCHAR(20),
    review_status VARCHAR(20),
    email_sent BOOLEAN,
    v1 FLOAT, v2 FLOAT, ..., v28 FLOAT
);
```

## Kiến trúc hệ thống

```
┌─────────────────┐
│  Web Browser    │
│  (Dashboard)    │
└────────┬────────┘
         │ HTTP/REST API
┌────────▼────────┐
│  Flask Server   │
│  (app.py)       │
├─────────────────┤
│ - Routes        │
│ - API Endpoints │
│ - ML Prediction │
│ - Email Service │
└────────┬────────┘
         │
    ┌────┴────┐
┌───▼───┐  ┌──▼──────┐
│SQLite │  │ML Model │
│  DB   │  │ (.pkl)  │
└───────┘  └─────────┘
```

## Yêu cầu đã hoàn thành

✅ RTFD-2.1: Alert Generation System
✅ RTFD-2.2: Alert Storage (Database)
✅ RTFD-2.3: Dashboard Visualization
✅ RTFD-2.4: Alert Filtering
✅ Email notification cho High Risk
✅ Real-time statistics
✅ Responsive UI design
✅ Dark/Light mode toggle

## Demo Data

Khi chạy lần đầu, database sẽ trống. Sử dụng tính năng **Transaction Fraud Check** để:
1. Generate random test data
2. Submit nhiều transactions
3. Dashboard sẽ hiển thị statistics và charts

## Troubleshooting

**Lỗi: Model file not found**
- Đảm bảo `fraud_detection_model.pkl` và `scaler.pkl` nằm cùng thư mục với `app.py`

**Lỗi: Database locked**
- Đóng tất cả connections cũ
- Xóa file `fraud_alerts.db` và restart

**Charts không hiển thị**
- Kiểm tra kết nối internet (Chart.js từ CDN)
- Mở Developer Console để xem errors

## Tác giả

Trịnh Gia Hiệp - Real-time Fraud Detection Module
Banking Data Analysis Platform - Report 16.11
