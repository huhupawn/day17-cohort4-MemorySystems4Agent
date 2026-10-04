# Báo Cáo Phân Tích & Benchmark Memory Systems (Day 17)

## 1. Kết Quả Benchmark Thực Nghiệm

Hệ thống được benchmark bằng bộ dữ liệu chuẩn tiếng Việt trên hai kịch bản: **Standard Benchmark** (10 hội thoại bình thường) và **Long-Context Stress Benchmark** (1 hội thoại gồm 16 lượt dài ép compaction).

### Bảng 1: Standard Benchmark (10 Hội Thoại, Cross-Session Recall)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 4,484 | 28,753 | 0.0% | 0.10 | 0 B | 0 |
| **Advanced Agent** | 5,568 | 31,908 | 100.0% | 1.00 | 328 B | 10 |

### Bảng 2: Long-Context Stress Benchmark (16 Lượt Dài, Compact Memory)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 827 | 26,430 | 0.0% | 0.10 | 0 B | 0 |
| **Advanced Agent** | 1,664 | 9,760 | 100.0% | 1.00 | 384 B | 28 |

---

## 2. Phân Tích Chuyên Sâu Các Trade-Off

### 2.1. Vì sao Advanced có Cross-session Recall vượt trội (100% vs 0%)?
- **Baseline Agent** chỉ duy trì short-term memory gắn với một `thread_id` duy nhất trong phiên làm việc. Khi chuyển sang một thread mới (các câu hỏi `recall_questions` được đặt ở thread độc lập), Baseline hoàn toàn không mang theo dữ liệu lịch sử và phải trả lời rỗng/từ chối do không biết người dùng là ai.
- **Advanced Agent** sở hữu lớp lưu trữ bền vững **`User.md`** thông qua `UserProfileStore`. Mọi thông tin quan trọng được trích xuất và cập nhật vào đĩa. Khi bất kỳ thread mới nào được khởi tạo, profile markdown được nạp vào context, giúp agent trả lời chính xác 100% các câu hỏi về tên, nghề nghiệp, nơi ở, món ăn, thức uống yêu thích và thú cưng.

### 2.2. Vì sao ở hội thoại ngắn, Advanced Agent lại tốn nhiều token hơn Baseline?
- Trong các hội thoại ngắn (như Standard Benchmark), kích thước ngữ cảnh hội thoại chưa đủ lớn để chạm ngưỡng nén.
- Lúc này, **Advanced Agent** luôn phải tải thêm toàn bộ nội dung của file `User.md` (và tóm tắt nếu có) vào mỗi lượt chat để đảm bảo agent nắm được bối cảnh người dùng.
- Chi phí cố định này (overhead) khiến `Prompt tokens processed` của Advanced cao hơn Baseline khoảng **10.9%** (31,908 so với 28,753 token). Đây là cái giá phải trả (trade-off) để đổi lấy khả năng ghi nhớ xuyên suốt phiên làm việc.

### 2.3. Sức mạnh của Compact Memory ở hội thoại dài (Giảm ~63% Prompt Tokens)
- Trong kịch bản **Stress Test** (16 lượt với ngữ cảnh kỹ thuật dài đặc), sự khác biệt bộc lộ rõ rệt:
  - **Baseline Agent** không có cơ chế nén, nó mang theo toàn bộ lịch sử thô tích lũy qua từng lượt: lượt 1 nạp lượt 1; lượt 2 nạp lượt 1 + 2; ... lượt 16 nạp toàn bộ 16 lượt cũ. Độ phức tạp prompt token tăng theo cấp số cộng quadratic $\mathcal{O}(N^2)$, làm tiêu tốn tới **26,430 prompt tokens**.
  - **Advanced Agent** kích hoạt `CompactMemoryManager`: khi số token trong thread vượt ngưỡng 400 token, agent tự động tóm tắt các message cũ (giữ lại 4 message gần nhất) thành một bản tóm tắt súc tích. Tổng prompt token được xử lý chỉ còn **9,760 prompt tokens** (tiết kiệm **63.1%** chi phí ngữ cảnh).
- Điều này chứng minh luận điểm quan trọng nhất: **Compact Memory không tối ưu agent tokens sinh ra, mà tối ưu trực tiếp Prompt Tokens Processed**, ngăn chặn hiện tượng context window explosion.

### 2.4. Memory Growth và Rủi Ro Tiềm Ẩn
- File `User.md` của `dungct` tăng từ 0 lên 328 Bytes (Standard) và 384 Bytes (Stress).
- **Rủi ro khi hệ thống chạy thực tế:**
  1. **Profile Bloat (Phình to file profile):** Nếu không có cơ chế chọn lọc, mọi câu nói vu vơ của người dùng đều bị đưa vào `User.md`, biến profile thành một tài liệu khổng lồ khiến prompt overhead tăng vọt.
  2. **Stale/Conflicting Facts (Dữ liệu lỗi thời/xung đột):** Người dùng thay đổi địa điểm hoặc nghề nghiệp, nếu chỉ append mà không ghi đè sẽ dẫn tới ảo giác (hallucination) khi model thấy hai thông tin mâu thuẫn trong prompt.

---

## 3. Các Tính Năng Bonus Đã Triển Khai (Mức 90-100 Điểm)

### 3.1. Structured Entity Extraction
Thay vì lưu chuỗi text tự do, hệ thống định nghĩa các entity có cấu trúc rõ ràng:
- `name`: Tên người dùng
- `location`: Địa điểm hiện tại
- `profession`: Nghề nghiệp
- `favorite_drink` / `favorite_food`: Sở thích ăn uống
- `pet`: Thú cưng
- `style`: Phong cách phản hồi mong muốn
- `interests`: Mối quan tâm kỹ thuật

### 3.2. Confidence Threshold & Lọc Nhiễu (Noise Filtering)
- **Chặn câu hỏi:** Các turn dạng hỏi ("Bạn có biết DũngCT không?", "?") có confidence thấp (< 0.7) nên không bao giờ bị ghi nhầm thành profile facts.
- **Loại bỏ trò đùa:** Phát hiện câu đùa ("đùa với đồng nghiệp... chuyển sang product manager") -> loại bỏ hoàn toàn, bảo toàn nghề nghiệp chính xác `MLOps engineer`.
- **Loại bỏ sự kiện ngắn hạn:** Phát hiện chuyến đi ngắn hạn ("Hà Nội chỉ là nơi mình vừa bay ra họp") -> không ghi đè Hà Nội vào trường nơi ở `location`.

### 3.3. Conflict & Correction Handling (Xử Lý Đính Chính)
- Khi người dùng đính chính:
  - Nơi ở: Đà Nẵng -> Huế (`conv-03`) -> Đà Nẵng (`stress-01`).
  - Nghề nghiệp: Backend engineer -> MLOps engineer (`conv-06`).
- Hệ thống nhận diện các marker đính chính (`"không còn làm"`, `"chuyển sang"`, `"giờ mình đang ở ... chứ không còn ở"`), tự động cập nhật giá trị mới nhất vào `User.md` và lưu vết vào phần `History & Corrections`, ngăn chặn triệt để xung đột dữ liệu.

### 3.4. Bounded Context Compaction & Memory Decay
- Bộ tóm tắt `summarize_messages` được thiết kế có chặn trên (bounded): khi compact lặp lại nhiều lần, nó gộp các chủ đề vào danh sách set duy nhất thay vì nhân bản dòng tóm tắt.
- Giữ kích thước tóm tắt luôn ổn định dưới 50 tokens dù trải qua 28 lần compaction liên tiếp.
