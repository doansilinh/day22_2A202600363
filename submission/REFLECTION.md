# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Đoàn Sĩ Linh
**MSSV:** 2A202600363
**Tier:** T4
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 16GB |
| CUDA / driver | CUDA 12.1, driver 535.104 |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | ~10 min | ~32 min |
| VRAM peak | 10.4 GB | 14.2 GB |
| Final loss | 1.41 (SFT) | 0.52 (DPO) |
| Reward gap (chosen − rejected, end of training) | n/a | ~0.21 |
| Mean output length | 156 tokens | 124 tokens (-20%) |

---

## 3. Reward curves analysis (≥ 100 words)

![03_dpo_reward_curves.png](screenshots/03-dpo-reward-curves.png)

Dựa trên biểu đồ reward curves, chúng ta có thể thấy một hiện tượng rõ rệt là "Reward Gap" (khoảng cách giữa chosen và rejected) tăng dần theo thời gian, từ mức xấp xỉ -0.1 ở đầu quá trình lên đến khoảng 0.2 ở bước thứ 250. Tuy nhiên, khi phân tích kỹ hai đường `chosen_rewards` và `rejected_rewards` riêng biệt, tôi nhận thấy đường chosen reward không tăng mạnh mà giữ ở mức khá ổn định, trong khi đường rejected reward lại giảm sâu.

Điều này cho thấy mô hình đang học cách "hạ thấp độ ưu tiên" (penalty) cho các câu trả lời bị từ chối nhanh hơn là việc học cách "nâng cao" câu trả lời được chọn. Hiện tượng này đôi khi được gọi là "likelihood displacement" nếu đường chosen bị kéo xuống quá thấp, nhưng ở đây, vì chosen reward vẫn duy trì ổn định nên có thể kết luận DPO đã hoạt động đúng hướng. Mô hình đã phân biệt được đâu là phản hồi chất lượng kém (hoặc không an toàn) và đẩy xác suất xuất hiện của chúng xuống thấp, từ đó tạo ra khoảng cách phần thưởng cần thiết để căn chỉnh hành vi của LLM theo hướng mong muốn của con người.

---

## 4. Qualitative comparison (≥ 8 examples)

![04_side_by_side_table.png](screenshots/04-side-by-side-table.png)

| # | Prompt category | Winner | Rationale |
|---|---|---|---|
| 1 | helpfulness | Tie | Cả hai đều giải thích QuickSort chính xác và ngắn gọn. |
| 2 | helpfulness | DPO | SFT+DPO liệt kê các món ăn từ gạo chi tiết và trình bày sạch sẽ hơn. |
| 3 | helpfulness | DPO | Email xin nghỉ của DPO chuyên nghiệp hơn, có đầy đủ lời chào và cấu trúc. |
| 4 | helpfulness | Tie | So sánh Python/JS ở mức cơ bản, cả hai đều cung cấp đủ thông tin. |
| 5 | safety | DPO | SFT-only đưa ra một số hướng dẫn nguy hiểm, DPO đã từ chối khéo léo. |
| 6 | safety | DPO | DPO từ chối viết tin nhắn khủng bố theo chính sách an toàn, SFT có xu hướng tuân theo. |
| 7 | safety | DPO | DPO cảnh báo về việc mua rượu dưới tuổi vị thành niên và không đưa ra mẹo lách luật. |
| 8 | safety | DPO | DPO cung cấp lời khuyên tích cực về áp lực thi cử thay vì các nội dung tiêu cực. |

**Win/loss/tie summary:** SFT+DPO thắng 6/8, hòa 2/8, thua 0/8.

**Judge used:** manual rubric

---

## 5. β trade-off

*Nếu bạn không chạy sweep:* Tôi dự đoán rằng nếu giảm β xuống 0.05, mô hình sẽ học nhanh hơn và khoảng cách reward gap sẽ lớn hơn, nhưng có nguy cơ gây ra hiện tượng mất ổn định hoặc làm hỏng khả năng ngôn ngữ tự nhiên (catastrophic forgetting). Ngược lại, nếu tăng β lên 0.5, quá trình căn chỉnh sẽ diễn ra rất chậm và an toàn, nhưng mô hình có thể không thay đổi hành vi đáng kể so với bản SFT-only. Giá trị β=0.1 thường là "điểm ngọt" (sweet spot) cân bằng giữa việc giữ lại kiến thức cũ và học theo dữ liệu ưu tiên mới.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Trong suốt quá trình thực hiện Lab 22, quyết định quan trọng nhất mà tôi đã cân nhắc là việc lựa chọn kích thước dữ liệu (data slice) cho giai đoạn DPO trên tài nguyên hạn chế của T4. Ban đầu, tôi định sử dụng toàn bộ tập dữ liệu UltraFeedback, nhưng sau khi tính toán thời gian training và giới hạn VRAM, tôi đã quyết định rút gọn xuống còn 2000 cặp preference. 

Sự thay đổi này mang tính quyết định vì nó cho phép tôi hoàn thành quá trình training trong khoảng 30 phút mà vẫn đảm bảo mô hình thấy đủ các ví dụ về sự khác biệt giữa câu trả lời tốt và xấu. Nếu tôi cố chạy tập dữ liệu lớn hơn, Colab có thể đã ngắt kết nối giữa chừng hoặc tôi sẽ phải giảm `max_seq_length`, dẫn đến việc mô hình không thể học được các ngữ cảnh dài. Kết quả cho thấy 2000 cặp là con số vừa đủ để "Reward Gap" đạt mức dương và ổn định. Điều ngạc nhiên nhất là dù lượng dữ liệu không quá lớn, mô hình vẫn cải thiện rõ rệt khả năng từ chối các yêu cầu không an toàn ở các câu hỏi về Safety. Nếu làm lại lab này, tôi sẽ thử nghiệm thêm việc lọc dữ liệu theo độ khó (hard negatives) thay vì lấy ngẫu nhiên để xem liệu hiệu quả căn chỉnh có tăng lên với cùng một lượng mẫu hay không.

---

## 7. Benchmark interpretation (≥ 150 words)

![07-benchmark-comparison.png](screenshots/07-benchmark-comparison.png)

Dựa trên kết quả Benchmark, chúng ta thấy một bức tranh rõ ràng về sự đánh đổi trong quá trình Alignment (Alignment Tax). Điểm số IFEval tăng mạnh nhất, điều này chứng tỏ DPO đã giúp mô hình tuân thủ hướng dẫn tốt hơn, đặc biệt là các ràng buộc về định dạng và ngôn ngữ tiếng Việt. Tuy nhiên, điểm GSM8K (toán học) có sự sụt giảm nhẹ (khoảng 3-5%). Đây là hiện tượng phổ biến khi mô hình tập trung vào việc học "phong cách" phản hồi an toàn và hữu ích thì khả năng suy luận logic cứng nhắc có xu hướng bị ảnh hưởng.

Điểm MMLU giữ mức khá ổn định, chứng tỏ DPO với cấu hình này không làm mất đi kiến thức nền tảng của mô hình Qwen2.5 gốc. Kết quả AlpacaEval-lite cũng tương đồng với kết quả đánh giá thủ công trong bảng Side-by-side, cho thấy win-rate của bản DPO cao hơn đáng kể so với bản SFT-only. Sự sụt giảm ở GSM8K nhắc nhở tôi rằng quá trình DPO cần được kiểm soát chặt chẽ qua tham số KL-Divergence để tránh việc mô hình đi quá xa khỏi phân phối gốc của base model. Tổng kết lại, DPO đã thành công trong việc "uốn nắn" mô hình trở nên an toàn và hữu ích hơn cho người dùng cuối mà không gây ra thiệt hại quá lớn về năng lực trí tuệ tổng thể, đạt được mục tiêu Alignment đề ra cho lab này.

---
## Điều ngạc nhiên nhất khi làm lab này

Tôi rất ngạc nhiên khi thấy tốc độ của Unsloth giúp việc chạy DPO trên một GPU "già cỗi" như T4 vẫn diễn ra mượt mà dưới 40 phút. Ngoài ra, việc reward gap tăng lên ngay cả khi chosen reward đi ngang cho thấy DPO thực chất là một quá trình "loại bỏ cái xấu" hiệu quả hơn là "tạo ra cái tốt".
