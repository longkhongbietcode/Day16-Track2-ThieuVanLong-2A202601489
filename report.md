# Báo cáo ngắn kết quả benchmark LightGBM

- Mô hình LightGBM được huấn luyện trên CPU 2 vCPU với thời gian 4,543301 giây.
- Thời gian tải bộ dữ liệu gồm 284.807 giao dịch là 2,341493 giây.
- Early stopping xác định vòng lặp tốt nhất là iteration 95.
- Mô hình đạt AUC-ROC 0,968085 và Accuracy 0,999526 trên tập test.
- F1-Score đạt 0,852459, cho thấy mô hình phát hiện gian lận khá hiệu quả trên dữ liệu mất cân bằng.
- Precision đạt 0,917647 và Recall đạt 0,795918, nghĩa là mô hình có độ chính xác cảnh báo cao nhưng vẫn bỏ sót một phần giao dịch gian lận.
- Inference latency cho một giao dịch là khoảng 1,227418 ms.
- Inference throughput cho batch 1.000 dòng đạt khoảng 291.138,83 dòng/giây, chứng tỏ LightGBM suy luận rất nhanh trên CPU.
