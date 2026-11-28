# muninn benchmark report

**dedicated GCP c4-standard-16 (16 vCPUs, Xeon Platinum 8581C), europe-west4-a, nothing else running; commit ac3828c**

- Generated: 2026-11-28 11:55 UTC
- Machine: c4-standard-16, INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, 16 CPUs available (pinned to all), 63,206.4 MB RAM
- CPU flags: sse4_2, avx, avx2, fma, f16c, avx512f, avx512bw, avx512vl, avx512dq, avx512_vnni, avx512_bf16, avx512_fp16, avx_vnni, amx_tile
- muninn commit ac3828c8c74ad6b01aa6b0285779c6df2fd0e952, rustc 1.91.0 (f8297e351 2025-10-28), RUSTFLAGS `-C target-cpu=native`, SIMD avx512

Engines: faiss-hnsw 1.16.0, hnswlib 0.8.0, muninn 0.1.0 (ac3828c8c74a), muninn-engine 0.1.0 (ac3828c8c74a), muninn-flat 0.1.0 (ac3828c8c74a), muninn-hnsw 0.1.0 (ac3828c8c74a), muninn-hnsw-sq8 0.1.0 (ac3828c8c74a), tantivy tantivy v0.27.0, index_format v7, usearch 2.26.6

## glove-100-angular

1,183,514 vectors, 100 dimensions, metric `cosine`, 10,000 queries, k = 10, 16 threads.

Ground truth check: muninn exact search on 1000 sampled queries matches the provided neighbours for 100.0% of queries by id, 100.0% counting distance ties (recall 1.0000).

### Build

| engine | version | params | build s | threads | index MB | RSS growth MB |
|---|---|---|---|---|---|---|
| muninn-hnsw | 0.1.0 (ac3828c8c74a) | M=16 efC=200 | 42.2 | 16 | 655.1 | 687.0 |
| muninn-hnsw-sq8 | 0.1.0 (ac3828c8c74a) | M=16 efC=200 Sq8 rerank=40 | 42.8 | 16 | 773.6 | 687.0 |
| muninn-engine | 0.1.0 (ac3828c8c74a) | M=16 efC=200 | 58.4 | 16 | 711.2 | 193.6 |
| hnswlib | 0.8.0 | M=16 efC=200 | 38.7 | 16 | 649.2 | 805.0 |
| faiss-hnsw | 1.16.0 | M=16 efC=200 | 35.1 | 16 | 644.2 | 695.2 |
| usearch | 2.26.6 | M=16 efC=200 | 47.6 | 16 | 649.2 | 805.0 |
| muninn-flat | 0.1.0 (ac3828c8c74a) | exact | 0.2 | 1 | 473.4 | 473.5 |

### Best QPS at a recall target (one thread / all threads)

Highest throughput among sweep points reaching at least the given recall@k; `ef` in brackets.

| engine | recall ≥ 0.8 | recall ≥ 0.9 | recall ≥ 0.95 | recall ≥ 0.98 | recall ≥ 0.99 |
|---|---|---|---|---|---|
| muninn-hnsw M=16 efC=200 | 6,799 (96) / 71,825 (96) | 1,944 (384) / 20,797 (384) | - | - | - |
| muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40 | 9,082 (96) / 101,558 (96) | 2,608 (384) / 29,151 (384) | - | - | - |
| muninn-engine M=16 efC=200 | 6,462 (96) / 68,541 (96) | 1,867 (384) / 20,080 (384) | - | - | - |
| hnswlib M=16 efC=200 | 5,925 (96) / 55,423 (96) | 1,718 (384) / 16,260 (384) | - | - | - |
| faiss-hnsw M=16 efC=200 | 6,115 (96) / 61,981 (96) | 1,676 (384) / 17,616 (384) | - | - | - |
| usearch M=16 efC=200 | 3,729 (96) / 39,978 (96) | 1,098 (384) / 11,743 (384) | - | - | - |

### muninn-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4891 | 38,279 | 396,632 | 24.7 | 39.6 | 48.3 | yes |
| 16 | 0.5765 | 27,775 | 288,123 | 34.3 | 53.4 | 63.6 | yes |
| 24 | 0.6456 | 20,720 | 213,378 | 46.0 | 71.0 | 83.7 | yes |
| 32 | 0.6900 | 16,667 | 175,272 | 57.6 | 86.3 | 102.6 | yes |
| 48 | 0.7448 | 12,121 | 127,660 | 79.9 | 115.8 | 137.5 | yes |
| 64 | 0.7797 | 9,601 | 101,093 | 102.1 | 142.4 | 168.3 | yes |
| 96 | 0.8208 | 6,799 | 71,825 | 145.9 | 195.5 | 226.5 | yes |
| 128 | 0.8445 | 5,323 | 56,328 | 187.5 | 245.1 | 276.9 | yes |
| 192 | 0.8765 | 3,715 | 39,488 | 271.3 | 345.1 | 380.7 | yes |
| 256 | 0.8966 | 2,858 | 30,395 | 353.3 | 447.9 | 485.9 | yes |
| 384 | 0.9202 | 1,944 | 20,797 | 520.2 | 663.9 | 706.2 | yes |
| 512 | 0.9345 | 1,487 | 15,862 | 681.1 | 876.1 | 929.8 | yes |

### muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7215 | 18,525 | 200,677 | 52.1 | 74.9 | 88.4 |  |
| 16 | 0.7215 | 18,737 | 202,839 | 51.6 | 74.3 | 87.0 | yes |
| 24 | 0.7215 | 18,446 | 201,920 | 52.4 | 75.6 | 88.3 |  |
| 32 | 0.7215 | 18,577 | 201,230 | 52.1 | 74.8 | 87.1 |  |
| 48 | 0.7448 | 16,598 | 175,817 | 58.5 | 82.5 | 97.9 | yes |
| 64 | 0.7806 | 13,084 | 139,719 | 74.8 | 103.5 | 121.2 | yes |
| 96 | 0.8202 | 9,082 | 101,558 | 109.3 | 144.4 | 166.1 | yes |
| 128 | 0.8449 | 6,929 | 79,176 | 144.3 | 187.1 | 212.1 | yes |
| 192 | 0.8767 | 4,883 | 55,530 | 205.4 | 263.4 | 290.3 | yes |
| 256 | 0.8963 | 3,818 | 42,701 | 264.2 | 335.8 | 368.0 | yes |
| 384 | 0.9202 | 2,608 | 29,151 | 387.4 | 496.0 | 532.1 | yes |
| 512 | 0.9345 | 1,996 | 22,045 | 503.8 | 655.4 | 700.5 | yes |

### muninn-engine M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4867 | 35,062 | 365,537 | 27.1 | 42.5 | 50.7 | yes |
| 16 | 0.5777 | 25,374 | 267,048 | 37.4 | 58.5 | 70.1 | yes |
| 24 | 0.6459 | 19,191 | 201,843 | 49.8 | 76.0 | 89.3 | yes |
| 32 | 0.6906 | 15,541 | 164,001 | 61.6 | 91.8 | 108.0 | yes |
| 48 | 0.7449 | 11,409 | 120,075 | 85.2 | 121.6 | 142.8 | yes |
| 64 | 0.7786 | 9,100 | 96,083 | 107.8 | 149.2 | 174.1 | yes |
| 96 | 0.8213 | 6,462 | 68,541 | 153.6 | 204.5 | 235.1 | yes |
| 128 | 0.8468 | 5,058 | 53,845 | 197.8 | 255.8 | 291.9 | yes |
| 192 | 0.8771 | 3,571 | 37,885 | 282.2 | 360.1 | 397.6 | yes |
| 256 | 0.8967 | 2,783 | 29,243 | 363.5 | 458.9 | 497.8 | yes |
| 384 | 0.9197 | 1,867 | 20,080 | 543.8 | 685.8 | 722.9 | yes |
| 512 | 0.9339 | 1,416 | 15,242 | 717.1 | 910.5 | 953.5 | yes |

### hnswlib M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4709 | 35,179 | 306,318 | 28.2 | 44.6 | 53.2 | yes |
| 16 | 0.5587 | 25,089 | 222,651 | 39.1 | 61.0 | 72.4 | yes |
| 24 | 0.6273 | 18,487 | 167,386 | 52.8 | 80.4 | 94.7 | yes |
| 32 | 0.6718 | 14,887 | 133,912 | 66.0 | 99.2 | 117.0 | yes |
| 48 | 0.7281 | 10,659 | 97,554 | 92.6 | 133.3 | 158.7 | yes |
| 64 | 0.7620 | 8,387 | 77,676 | 117.7 | 164.0 | 195.0 | yes |
| 96 | 0.8051 | 5,925 | 55,423 | 168.2 | 225.0 | 260.9 | yes |
| 128 | 0.8315 | 4,645 | 43,332 | 217.0 | 282.1 | 321.5 | yes |
| 192 | 0.8626 | 3,208 | 30,412 | 316.8 | 394.7 | 438.1 | yes |
| 256 | 0.8837 | 2,473 | 23,444 | 411.7 | 504.2 | 552.0 | yes |
| 384 | 0.9095 | 1,718 | 16,260 | 594.1 | 717.7 | 762.6 | yes |
| 512 | 0.9256 | 1,314 | 12,468 | 776.2 | 934.5 | 981.7 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 1.4..4.6 us.

### faiss-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4758 | 37,369 | 355,481 | 29.2 | 46.5 | 57.3 | yes |
| 16 | 0.5629 | 26,656 | 253,897 | 39.5 | 62.5 | 75.1 | yes |
| 24 | 0.6313 | 19,548 | 186,697 | 52.3 | 80.5 | 97.1 | yes |
| 32 | 0.6755 | 15,591 | 149,688 | 65.4 | 99.6 | 118.2 | yes |
| 48 | 0.7315 | 11,154 | 109,750 | 90.1 | 131.3 | 156.8 | yes |
| 64 | 0.7639 | 8,662 | 86,915 | 114.6 | 161.8 | 189.0 | yes |
| 96 | 0.8064 | 6,115 | 61,981 | 164.9 | 221.7 | 258.0 | yes |
| 128 | 0.8322 | 4,696 | 48,269 | 215.7 | 281.2 | 323.4 | yes |
| 192 | 0.8628 | 3,210 | 33,631 | 321.3 | 403.8 | 449.1 | yes |
| 256 | 0.8827 | 2,474 | 25,870 | 429.6 | 530.0 | 578.4 | yes |
| 384 | 0.9067 | 1,676 | 17,616 | 610.2 | 727.5 | 774.8 | yes |
| 512 | 0.9212 | 1,253 | 13,363 | 818.1 | 975.5 | 1,044.9 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 3.0..24.8 us.

### usearch M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4623 | 20,921 | 223,269 | 53.6 | 84.0 | 101.7 | yes |
| 16 | 0.5530 | 14,946 | 161,274 | 72.4 | 113.3 | 134.0 | yes |
| 24 | 0.6214 | 11,146 | 119,541 | 94.4 | 145.1 | 173.1 | yes |
| 32 | 0.6655 | 9,005 | 96,264 | 116.4 | 176.7 | 209.4 | yes |
| 48 | 0.7234 | 6,551 | 69,642 | 157.8 | 233.1 | 274.0 | yes |
| 64 | 0.7593 | 5,194 | 55,431 | 198.1 | 283.3 | 334.0 | yes |
| 96 | 0.8030 | 3,729 | 39,978 | 276.4 | 381.5 | 443.4 | yes |
| 128 | 0.8297 | 2,917 | 31,288 | 354.6 | 474.7 | 540.6 | yes |
| 192 | 0.8630 | 2,052 | 21,971 | 508.2 | 657.1 | 737.2 | yes |
| 256 | 0.8840 | 1,582 | 16,856 | 660.1 | 841.4 | 921.6 | yes |
| 384 | 0.9097 | 1,098 | 11,743 | 945.7 | 1,195.8 | 1,281.0 | yes |
| 512 | 0.9248 | 855 | 9,024 | 1,228.0 | 1,559.9 | 1,657.3 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 8.7..41.1 us.

### muninn-flat exact

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| - | 0.9999 | 35 | 326 | 28,351.9 | 28,649.1 | 28,827.1 | yes |

## nytimes-256-angular

290,000 vectors, 256 dimensions, metric `cosine`, 10,000 queries, k = 10, 16 threads.

Ground truth check: muninn exact search on 1000 sampled queries matches the provided neighbours for 92.0% of queries by id, 100.0% counting distance ties (recall 0.9871).

### Build

| engine | version | params | build s | threads | index MB | RSS growth MB |
|---|---|---|---|---|---|---|
| muninn-hnsw | 0.1.0 (ac3828c8c74a) | M=16 efC=200 | 16.5 | 16 | 341.5 | 361.7 |
| muninn-hnsw-sq8 | 0.1.0 (ac3828c8c74a) | M=16 efC=200 Sq8 rerank=40 | 16.9 | 16 | 415.8 | 361.7 |
| muninn-engine | 0.1.0 (ac3828c8c74a) | M=16 efC=200 | 25.1 | 16 | 355.4 | 123.1 |
| hnswlib | 0.8.0 | M=16 efC=200 | 11.9 | 16 | 340.0 | 381.5 |
| faiss-hnsw | 1.16.0 | M=16 efC=200 | 12.8 | 16 | 338.8 | 354.7 |
| usearch | 2.26.6 | M=16 efC=200 | 17.8 | 16 | 340.0 | 391.6 |
| muninn-flat | 0.1.0 (ac3828c8c74a) | exact | 0.1 | 1 | 297.0 | 297.2 |

### Best QPS at a recall target (one thread / all threads)

Highest throughput among sweep points reaching at least the given recall@k; `ef` in brackets.

| engine | recall ≥ 0.8 | recall ≥ 0.9 | recall ≥ 0.95 | recall ≥ 0.98 | recall ≥ 0.99 |
|---|---|---|---|---|---|
| muninn-hnsw M=16 efC=200 | 13,994 (32) / 135,506 (32) | 3,176 (192) / 31,933 (192) | - | - | - |
| muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40 | 20,066 (32) / 207,904 (24) | 5,476 (192) / 57,882 (192) | - | - | - |
| muninn-engine M=16 efC=200 | 13,627 (32) / 133,354 (32) | 3,116 (192) / 31,625 (192) | - | - | - |
| hnswlib M=16 efC=200 | 8,684 (48) / 79,575 (48) | 2,146 (256) / 19,408 (256) | - | - | - |
| faiss-hnsw M=16 efC=200 | 8,595 (48) / 85,583 (48) | 1,968 (256) / 20,436 (256) | - | - | - |
| usearch M=16 efC=200 | 4,716 (48) / 48,586 (48) | 1,168 (256) / 11,976 (256) | - | - | - |

### muninn-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.6304 | 31,562 | 303,407 | 30.2 | 48.5 | 58.0 | yes |
| 16 | 0.7145 | 22,982 | 218,735 | 41.3 | 66.5 | 79.1 | yes |
| 24 | 0.7715 | 17,106 | 166,322 | 55.8 | 88.5 | 103.8 | yes |
| 32 | 0.8005 | 13,994 | 135,506 | 68.6 | 106.7 | 126.5 | yes |
| 48 | 0.8347 | 10,246 | 100,010 | 94.8 | 142.0 | 166.5 | yes |
| 64 | 0.8561 | 8,173 | 80,582 | 119.9 | 173.9 | 205.8 | yes |
| 96 | 0.8784 | 5,825 | 58,133 | 170.7 | 235.5 | 276.9 | yes |
| 128 | 0.8905 | 4,566 | 45,604 | 220.6 | 293.0 | 337.1 | yes |
| 192 | 0.9064 | 3,176 | 31,933 | 321.1 | 409.6 | 449.3 | yes |
| 256 | 0.9162 | 2,430 | 24,508 | 423.5 | 529.5 | 573.8 | yes |
| 384 | 0.9298 | 1,641 | 16,644 | 633.0 | 775.1 | 816.5 | yes |
| 512 | 0.9396 | 1,231 | 12,502 | 849.2 | 1,027.9 | 1,072.4 | yes |

### muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.8209 | 20,019 | 205,708 | 48.4 | 71.0 | 83.0 |  |
| 16 | 0.8209 | 20,029 | 207,454 | 48.4 | 70.8 | 82.5 |  |
| 24 | 0.8209 | 19,949 | 207,904 | 48.6 | 71.3 | 82.9 |  |
| 32 | 0.8209 | 20,066 | 206,951 | 48.3 | 71.1 | 82.7 | yes |
| 48 | 0.8346 | 17,500 | 181,503 | 55.7 | 80.3 | 92.9 | yes |
| 64 | 0.8557 | 14,009 | 145,399 | 70.0 | 99.4 | 115.0 | yes |
| 96 | 0.8775 | 10,001 | 104,967 | 99.0 | 134.7 | 156.7 | yes |
| 128 | 0.8900 | 7,893 | 82,719 | 127.0 | 167.9 | 190.7 | yes |
| 192 | 0.9061 | 5,476 | 57,882 | 185.3 | 236.8 | 260.1 | yes |
| 256 | 0.9163 | 4,210 | 44,712 | 242.8 | 305.3 | 328.3 | yes |
| 384 | 0.9299 | 2,848 | 30,139 | 361.5 | 447.3 | 472.5 | yes |
| 512 | 0.9394 | 2,133 | 22,498 | 485.8 | 594.4 | 623.5 | yes |

### muninn-engine M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.6315 | 30,064 | 291,020 | 31.6 | 50.5 | 59.1 | yes |
| 16 | 0.7143 | 22,072 | 215,166 | 43.2 | 68.3 | 81.1 | yes |
| 24 | 0.7730 | 16,652 | 162,431 | 57.2 | 90.0 | 108.9 | yes |
| 32 | 0.8015 | 13,627 | 133,354 | 70.4 | 109.3 | 128.2 | yes |
| 48 | 0.8372 | 10,010 | 98,658 | 96.9 | 145.0 | 172.4 | yes |
| 64 | 0.8562 | 7,977 | 79,146 | 122.7 | 177.1 | 207.5 | yes |
| 96 | 0.8775 | 5,718 | 57,397 | 173.9 | 238.7 | 275.0 | yes |
| 128 | 0.8910 | 4,482 | 45,166 | 224.6 | 297.2 | 334.3 | yes |
| 192 | 0.9072 | 3,116 | 31,625 | 328.2 | 417.5 | 456.2 | yes |
| 256 | 0.9173 | 2,394 | 24,325 | 430.4 | 534.9 | 573.7 | yes |
| 384 | 0.9315 | 1,622 | 16,550 | 640.2 | 784.7 | 825.5 | yes |
| 512 | 0.9408 | 1,228 | 12,447 | 848.4 | 1,030.6 | 1,077.1 | yes |

### hnswlib M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.5820 | 28,798 | 247,569 | 34.5 | 55.8 | 65.8 | yes |
| 16 | 0.6715 | 20,060 | 176,639 | 48.4 | 77.5 | 91.1 | yes |
| 24 | 0.7355 | 14,861 | 133,653 | 65.7 | 104.2 | 120.8 | yes |
| 32 | 0.7723 | 11,890 | 105,548 | 82.5 | 128.5 | 151.6 | yes |
| 48 | 0.8126 | 8,684 | 79,575 | 114.4 | 170.4 | 198.5 | yes |
| 64 | 0.8374 | 6,898 | 63,018 | 144.8 | 209.2 | 242.8 | yes |
| 96 | 0.8621 | 4,885 | 45,793 | 206.8 | 284.1 | 329.9 | yes |
| 128 | 0.8780 | 3,843 | 35,764 | 264.4 | 348.7 | 399.2 | yes |
| 192 | 0.8945 | 2,687 | 25,254 | 383.4 | 482.7 | 535.6 | yes |
| 256 | 0.9057 | 2,146 | 19,408 | 491.3 | 605.6 | 653.5 | yes |
| 384 | 0.9211 | 1,439 | 13,404 | 723.7 | 865.0 | 908.6 | yes |
| 512 | 0.9316 | 1,093 | 10,137 | 984.4 | 1,148.5 | 1,195.6 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 1.3..30.7 us.

### faiss-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.6081 | 28,406 | 273,571 | 37.4 | 59.2 | 69.1 | yes |
| 16 | 0.6922 | 20,350 | 192,608 | 50.8 | 80.1 | 94.9 | yes |
| 24 | 0.7523 | 14,817 | 144,344 | 68.7 | 107.8 | 126.4 | yes |
| 32 | 0.7840 | 11,790 | 116,719 | 85.4 | 132.3 | 155.5 | yes |
| 48 | 0.8216 | 8,595 | 85,583 | 118.2 | 176.6 | 206.2 | yes |
| 64 | 0.8428 | 6,775 | 68,127 | 150.5 | 215.7 | 254.9 | yes |
| 96 | 0.8665 | 4,789 | 49,061 | 215.5 | 294.8 | 343.0 | yes |
| 128 | 0.8809 | 3,718 | 38,298 | 282.7 | 371.2 | 424.3 | yes |
| 192 | 0.8980 | 2,544 | 26,817 | 407.6 | 508.5 | 553.7 | yes |
| 256 | 0.9085 | 1,968 | 20,436 | 530.8 | 642.4 | 691.6 | yes |
| 384 | 0.9233 | 1,336 | 13,820 | 787.3 | 927.0 | 976.5 | yes |
| 512 | 0.9327 | 1,004 | 10,374 | 1,054.2 | 1,220.3 | 1,277.8 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 4.2..21.4 us.

### usearch M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.5867 | 14,739 | 150,246 | 73.5 | 118.0 | 138.5 | yes |
| 16 | 0.6774 | 10,559 | 107,674 | 100.2 | 158.9 | 186.0 | yes |
| 24 | 0.7382 | 7,903 | 81,129 | 131.0 | 205.9 | 241.8 | yes |
| 32 | 0.7739 | 6,388 | 65,905 | 160.4 | 250.7 | 295.8 | yes |
| 48 | 0.8142 | 4,716 | 48,586 | 217.6 | 331.3 | 396.8 | yes |
| 64 | 0.8347 | 3,793 | 38,947 | 272.4 | 405.2 | 479.7 | yes |
| 96 | 0.8606 | 2,747 | 28,095 | 377.9 | 534.7 | 626.6 | yes |
| 128 | 0.8746 | 2,164 | 22,097 | 482.2 | 656.5 | 756.7 | yes |
| 192 | 0.8932 | 1,516 | 15,565 | 692.1 | 899.7 | 1,007.8 | yes |
| 256 | 0.9050 | 1,168 | 11,976 | 904.3 | 1,140.1 | 1,242.4 | yes |
| 384 | 0.9202 | 796 | 8,237 | 1,332.2 | 1,622.7 | 1,708.4 | yes |
| 512 | 0.9308 | 602 | 6,241 | 1,775.4 | 2,118.3 | 2,220.9 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 9.6..32.4 us.

### muninn-flat exact

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| - | 0.9848 | 78 | 609 | 12,909.4 | 13,104.2 | 13,243.6 | yes |

## sift-128-euclidean

1,000,000 vectors, 128 dimensions, metric `l2`, 10,000 queries, k = 10, 16 threads.

Ground truth check: muninn exact search on 1000 sampled queries matches the provided neighbours for 99.2% of queries by id, 100.0% counting distance ties (recall 0.9992).

### Build

| engine | version | params | build s | threads | index MB | RSS growth MB |
|---|---|---|---|---|---|---|
| muninn-hnsw | 0.1.0 (ac3828c8c74a) | M=16 efC=200 | 21.9 | 16 | 665.5 | 692.0 |
| muninn-hnsw-sq8 | 0.1.0 (ac3828c8c74a) | M=16 efC=200 Sq8 rerank=40 | 22.6 | 16 | 797.7 | 692.0 |
| muninn-engine | 0.1.0 (ac3828c8c74a) | M=16 efC=200 | 39.7 | 16 | 718.1 | 193.1 |
| hnswlib | 0.8.0 | M=16 efC=200 | 27.5 | 16 | 660.6 | 796.9 |
| faiss-hnsw | 1.16.0 | M=16 efC=200 | 28.0 | 16 | 656.3 | 696.2 |
| usearch | 2.26.6 | M=16 efC=200 | 38.4 | 16 | 660.5 | 763.3 |
| muninn-flat | 0.1.0 (ac3828c8c74a) | exact | 0.1 | 1 | 512.0 | 512.2 |

### Best QPS at a recall target (one thread / all threads)

Highest throughput among sweep points reaching at least the given recall@k; `ef` in brackets.

| engine | recall ≥ 0.8 | recall ≥ 0.9 | recall ≥ 0.95 | recall ≥ 0.98 | recall ≥ 0.99 |
|---|---|---|---|---|---|
| muninn-hnsw M=16 efC=200 | 35,837 (16) / 367,438 (16) | 21,782 (32) / 223,061 (32) | 15,806 (48) / 164,146 (48) | 8,894 (96) / 92,250 (96) | 6,920 (128) / 71,800 (128) |
| muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40 | 25,823 (10) / 282,603 (16) | 25,823 (10) / 282,603 (16) | 22,465 (48) / 246,086 (48) | 12,766 (96) / 139,714 (96) | 10,036 (128) / 110,341 (128) |
| muninn-engine M=16 efC=200 | 33,524 (16) / 349,930 (16) | 20,847 (32) / 215,840 (32) | 15,214 (48) / 157,760 (48) | 8,506 (96) / 88,143 (96) | 6,628 (128) / 69,349 (128) |
| hnswlib M=16 efC=200 | 25,541 (16) / 223,472 (16) | 15,450 (32) / 135,262 (32) | 8,644 (64) / 77,839 (64) | 6,151 (96) / 55,366 (96) | 3,343 (192) / 30,493 (192) |
| faiss-hnsw M=16 efC=200 | 30,231 (16) / 265,280 (16) | 17,912 (32) / 157,515 (32) | 10,172 (64) / 89,752 (64) | 7,111 (96) / 62,962 (96) | 5,199 (128) / 49,040 (128) |
| usearch M=16 efC=200 | 15,497 (16) / 154,766 (16) | 9,399 (32) / 93,231 (32) | 5,401 (64) / 53,381 (64) | 3,844 (96) / 37,866 (96) | 2,128 (192) / 20,775 (192) |

### muninn-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7236 | 47,864 | 497,212 | 20.3 | 29.1 | 33.9 | yes |
| 16 | 0.8147 | 35,837 | 367,438 | 27.5 | 37.6 | 44.4 | yes |
| 24 | 0.8787 | 27,097 | 277,680 | 36.8 | 48.2 | 54.6 | yes |
| 32 | 0.9146 | 21,782 | 223,061 | 46.2 | 58.9 | 65.5 | yes |
| 48 | 0.9518 | 15,806 | 164,146 | 64.3 | 79.2 | 86.2 | yes |
| 64 | 0.9694 | 12,510 | 129,885 | 81.7 | 100.0 | 107.2 | yes |
| 96 | 0.9855 | 8,894 | 92,250 | 115.1 | 140.9 | 150.5 | yes |
| 128 | 0.9916 | 6,920 | 71,800 | 148.2 | 181.5 | 192.2 | yes |
| 192 | 0.9962 | 4,859 | 50,482 | 211.1 | 260.3 | 275.7 | yes |
| 256 | 0.9979 | 3,735 | 39,286 | 275.8 | 338.1 | 357.5 | yes |
| 384 | 0.9990 | 2,601 | 27,630 | 395.4 | 489.6 | 518.4 | yes |
| 512 | 0.9992 | 2,019 | 21,530 | 508.2 | 633.1 | 672.8 | yes |

### muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.9368 | 25,823 | 282,562 | 39.0 | 47.0 | 52.1 | yes |
| 16 | 0.9368 | 25,800 | 282,603 | 39.0 | 47.0 | 51.7 |  |
| 24 | 0.9368 | 25,725 | 282,597 | 39.2 | 47.1 | 51.9 |  |
| 32 | 0.9368 | 25,787 | 282,503 | 39.0 | 47.0 | 51.7 |  |
| 48 | 0.9516 | 22,465 | 246,086 | 45.0 | 53.7 | 58.8 | yes |
| 64 | 0.9694 | 17,964 | 196,083 | 56.7 | 66.8 | 72.0 | yes |
| 96 | 0.9856 | 12,766 | 139,714 | 80.1 | 94.4 | 100.0 | yes |
| 128 | 0.9916 | 10,036 | 110,341 | 101.9 | 120.8 | 127.2 | yes |
| 192 | 0.9962 | 7,058 | 77,792 | 145.4 | 172.0 | 180.9 | yes |
| 256 | 0.9979 | 5,500 | 60,690 | 186.7 | 222.4 | 233.2 | yes |
| 384 | 0.9990 | 3,852 | 42,137 | 265.4 | 322.8 | 342.3 | yes |
| 512 | 0.9992 | 3,000 | 32,586 | 341.2 | 418.9 | 443.2 | yes |

### muninn-engine M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7244 | 44,229 | 465,848 | 22.0 | 30.9 | 35.9 | yes |
| 16 | 0.8160 | 33,524 | 349,930 | 29.4 | 39.7 | 45.6 | yes |
| 24 | 0.8793 | 25,622 | 265,738 | 38.9 | 50.5 | 57.0 | yes |
| 32 | 0.9153 | 20,847 | 215,840 | 48.3 | 60.6 | 67.7 | yes |
| 48 | 0.9519 | 15,214 | 157,760 | 66.8 | 81.4 | 88.3 | yes |
| 64 | 0.9696 | 12,005 | 125,101 | 85.0 | 102.7 | 110.2 | yes |
| 96 | 0.9853 | 8,506 | 88,143 | 120.5 | 144.8 | 154.3 | yes |
| 128 | 0.9919 | 6,628 | 69,349 | 154.7 | 186.5 | 196.8 | yes |
| 192 | 0.9964 | 4,611 | 48,344 | 223.0 | 269.8 | 283.6 | yes |
| 256 | 0.9979 | 3,581 | 37,865 | 287.4 | 349.2 | 366.9 | yes |
| 384 | 0.9989 | 2,490 | 26,552 | 413.0 | 507.1 | 533.7 | yes |
| 512 | 0.9993 | 1,928 | 20,568 | 532.8 | 657.7 | 696.2 | yes |

### hnswlib M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7086 | 34,498 | 303,378 | 29.6 | 42.2 | 49.6 | yes |
| 16 | 0.8017 | 25,541 | 223,472 | 40.0 | 54.7 | 63.5 | yes |
| 24 | 0.8648 | 19,208 | 168,517 | 53.7 | 70.1 | 79.9 | yes |
| 32 | 0.9032 | 15,450 | 135,262 | 66.9 | 84.7 | 94.1 | yes |
| 48 | 0.9426 | 11,072 | 98,618 | 93.6 | 114.2 | 124.6 | yes |
| 64 | 0.9634 | 8,644 | 77,839 | 120.3 | 144.7 | 155.6 | yes |
| 96 | 0.9814 | 6,151 | 55,366 | 169.2 | 202.7 | 215.3 | yes |
| 128 | 0.9893 | 4,777 | 43,355 | 219.2 | 261.7 | 275.8 | yes |
| 192 | 0.9953 | 3,343 | 30,493 | 313.2 | 375.7 | 394.8 | yes |
| 256 | 0.9972 | 2,593 | 23,738 | 402.1 | 484.9 | 510.7 | yes |
| 384 | 0.9988 | 1,839 | 16,723 | 570.6 | 692.9 | 729.2 | yes |
| 512 | 0.9990 | 1,459 | 12,997 | 710.7 | 873.0 | 920.9 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 1.4..10.2 us.

### faiss-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7157 | 41,099 | 357,301 | 27.1 | 38.5 | 46.6 | yes |
| 16 | 0.8075 | 30,231 | 265,280 | 36.1 | 48.9 | 59.3 | yes |
| 24 | 0.8735 | 22,383 | 195,170 | 48.5 | 63.0 | 73.3 | yes |
| 32 | 0.9108 | 17,912 | 157,515 | 59.6 | 75.1 | 85.0 | yes |
| 48 | 0.9485 | 12,800 | 113,904 | 83.7 | 101.2 | 112.1 | yes |
| 64 | 0.9675 | 10,172 | 89,752 | 104.0 | 123.7 | 132.8 | yes |
| 96 | 0.9835 | 7,111 | 62,962 | 152.3 | 179.6 | 190.1 | yes |
| 128 | 0.9904 | 5,199 | 49,040 | 204.7 | 238.7 | 250.6 | yes |
| 192 | 0.9955 | 3,526 | 34,228 | 299.9 | 348.2 | 362.5 | yes |
| 256 | 0.9974 | 2,750 | 26,203 | 393.5 | 456.5 | 476.0 | yes |
| 384 | 0.9987 | 1,843 | 18,245 | 569.0 | 661.2 | 687.1 | yes |
| 512 | 0.9990 | 1,425 | 14,168 | 735.4 | 861.6 | 900.8 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 3.7..18.7 us.

### usearch M=16 efC=200

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7104 | 20,996 | 208,197 | 55.4 | 77.1 | 91.6 | yes |
| 16 | 0.8007 | 15,497 | 154,766 | 72.7 | 99.2 | 115.1 | yes |
| 24 | 0.8642 | 11,639 | 117,000 | 95.5 | 124.8 | 141.5 | yes |
| 32 | 0.9022 | 9,399 | 93,231 | 116.8 | 147.8 | 164.7 | yes |
| 48 | 0.9427 | 6,853 | 67,922 | 159.5 | 196.1 | 216.4 | yes |
| 64 | 0.9635 | 5,401 | 53,381 | 201.2 | 244.2 | 262.9 | yes |
| 96 | 0.9814 | 3,844 | 37,866 | 280.1 | 339.4 | 360.4 | yes |
| 128 | 0.9889 | 3,013 | 29,743 | 357.3 | 434.1 | 459.0 | yes |
| 192 | 0.9944 | 2,128 | 20,775 | 505.6 | 619.6 | 654.0 | yes |
| 256 | 0.9969 | 1,662 | 16,089 | 647.3 | 798.3 | 840.6 | yes |
| 384 | 0.9983 | 1,163 | 11,346 | 911.5 | 1,139.1 | 1,202.9 | yes |
| 512 | 0.9987 | 905 | 8,781 | 1,165.4 | 1,464.4 | 1,549.4 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 9.2..22.8 us.

### muninn-flat exact

| ef | recall@k | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| - | 0.9994 | 33 | 319 | 30,178.9 | 30,534.0 | 30,875.4 | yes |

## MS MARCO passages, BM25 top-k

### Analyzer `default`

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 16 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn | 0.1.0 (ac3828c8c74a) | 1 | 6.8 | 145,990 | 83.2 | 1,990 | 19,695 | 367.0 | 1,400.5 | 2,250.7 | 10.00 |
| tantivy | tantivy v0.27.0, index_format v7 | 1 | 7.0 | 142,302 | 74.1 | 1,089 | 12,023 | 537.3 | 2,971.2 | 5,547.5 | 10.00 |

- muninn: 6.3 s indexing + 0.6 s writing; analyzer: alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped
- tantivy indexing with 16 threads: 4.6 s (215,866 docs/s; 2.6 s indexing + 2.1 s merging), 75.0 MB
- tantivy: 7.0 s indexing + 0.0 s merging; analyzer: SimpleTokenizer, LowerCaser, RemoveLongFilter(256) (tokens of at most 255 bytes kept)
- Result agreement muninn vs tantivy: mean overlap@10 0.9997, identical top-10 sets 99.8%, same top-1 99.9%.

### Analyzer `default`, end to end through the engine

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 16 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn-engine | 0.1.0 (ac3828c8c74a) | 1 | 16.7 | 60,030 | 474.2 | 1,947 | 19,386 | 377.9 | 1,420.1 | 2,281.4 | 10.00 |
| muninn | 0.1.0 (ac3828c8c74a) | 1 | 6.7 | 148,485 | 83.2 | 1,988 | 19,764 | 368.7 | 1,401.5 | 2,261.1 | 10.00 |

- muninn-engine index size is the whole live collection, including the stored passage text, id map and write-ahead log.
- muninn-engine: 10.4 s indexing + 6.3 s flush + compaction; analyzer: alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped
- muninn: 6.2 s indexing + 0.5 s writing; analyzer: alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped
- Result agreement muninn-engine vs muninn: mean overlap@10 1.0000, identical top-10 sets 100.0%, same top-1 100.0%.

### Analyzer `english`

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 16 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn | 0.1.0 (ac3828c8c74a) | 1 | 8.7 | 114,860 | 67.2 | 2,444 | 24,019 | 293.1 | 1,160.6 | 1,850.9 | 10.00 |
| tantivy | tantivy v0.27.0, index_format v7 | 1 | 17.4 | 57,324 | 60.2 | 1,576 | 17,349 | 325.9 | 2,261.6 | 4,183.4 | 10.00 |

- muninn: 8.2 s indexing + 0.5 s writing; analyzer: alphanumeric runs, Unicode lowercase, Latin diacritic folding, tokens > 255 bytes dropped, Lucene English stop words, Porter stemmer
- tantivy indexing with 16 threads: 3.8 s (263,730 docs/s; 2.2 s indexing + 1.6 s merging), 61.0 MB
- tantivy: 17.4 s indexing + 0.0 s merging; analyzer: SimpleTokenizer, LowerCaser, AsciiFoldingFilter, RemoveLongFilter(256), StopWordFilter(English, Lucene list), Stemmer(English, Snowball)
- Result agreement muninn vs tantivy: mean overlap@10 0.9798, identical top-10 sets 91.5%, same top-1 97.7%.

### Analyzer `english`, end to end through the engine

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 16 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 16t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn-engine | 0.1.0 (ac3828c8c74a) | 1 | 17.2 | 58,273 | 458.3 | 2,398 | 23,486 | 301.1 | 1,173.7 | 1,873.9 | 10.00 |
| muninn | 0.1.0 (ac3828c8c74a) | 1 | 8.6 | 116,399 | 67.2 | 2,447 | 24,023 | 292.2 | 1,164.2 | 1,850.3 | 10.00 |

- muninn-engine index size is the whole live collection, including the stored passage text, id map and write-ahead log.
- muninn-engine: 12.4 s indexing + 4.8 s flush + compaction; analyzer: alphanumeric runs, Unicode lowercase, Latin diacritic folding, tokens > 255 bytes dropped, Lucene English stop words, Porter stemmer
- muninn: 8.2 s indexing + 0.4 s writing; analyzer: alphanumeric runs, Unicode lowercase, Latin diacritic folding, tokens > 255 bytes dropped, Lucene English stop words, Porter stemmer
- Result agreement muninn-engine vs muninn: mean overlap@10 1.0000, identical top-10 sets 100.0%, same top-1 100.0%.

