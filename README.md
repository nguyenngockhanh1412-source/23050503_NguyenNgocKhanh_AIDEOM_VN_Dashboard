# AIDEOM-VN Dashboard

AIDEOM-VN Dashboard là web app phân tích các mô hình ra quyết định trong bối cảnh phát triển kinh tế Việt Nam trong kỷ nguyên AI. Hệ thống tích hợp các mô hình dự báo, tối ưu hóa, xếp hạng ưu tiên, mô phỏng lao động, phân tích rủi ro, học tăng cường và dashboard kịch bản trong một giao diện Streamlit thống nhất.

## 1. Mục tiêu

- Tổ chức 12 module mô hình hóa theo cùng một giao diện.
- Chuyển dữ liệu, tham số và thuật toán Python thành bảng kết quả và biểu đồ trực quan.
- Hỗ trợ so sánh phương án, phân tích ràng buộc và nhận diện các đánh đổi chính sách.

## 2. Cấu trúc hệ thống

- `app.py`: trang chủ và tổng quan dashboard.
- `pages/`: 12 module phân tích.
- `data/`: dữ liệu đầu vào dạng CSV.
- `utils/`: thành phần giao diện dùng chung.
- `.streamlit/config.toml`: cấu hình giao diện Streamlit.
- `requirements.txt`: danh sách thư viện cần cài đặt.

## 3. Cách chạy

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Chạy web app:

```bash
streamlit run app.py
```

Sau khi chạy, mở địa chỉ hiển thị trong terminal để xem dashboard trên trình duyệt.

## 4. Ghi chú dữ liệu

Các kết quả trong dashboard là kết quả mô phỏng theo bộ dữ liệu, tham số và kịch bản đã thiết lập trong hệ thống. Kết quả có giá trị hỗ trợ phân tích và so sánh phương án, nhưng không thay thế quy trình thẩm định chính sách hoặc dự báo kinh tế chính thức.
