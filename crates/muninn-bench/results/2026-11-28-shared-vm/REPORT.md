# muninn benchmark report

**Shared build VM (other agents compiling on other cores): numbers are NOISY, for pipeline validation only. Competitors from the first run (same day, same settings); muninn component runs at bcdd9f7, engine runs at 7b12b34 (component crates unchanged between the two).**

- Generated: 2026-11-28 02:11 UTC
- Machine: c4-standard-16, INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, 4 CPUs available (pinned to 0,1,8,9), 63,206.4 MB RAM
- CPU flags: sse4_2, avx, avx2, fma, f16c, avx512f, avx512bw, avx512vl, avx512dq, avx512_vnni, avx512_bf16, avx512_fp16, avx_vnni, amx_tile
- muninn commit 7b12b34, rustc 1.91.0 (f8297e351 2025-10-28), RUSTFLAGS `-C target-cpu=native`, SIMD avx512

Engines: faiss-hnsw 1.16.0, hnswlib 0.8.0, muninn 0.1.0 (7b12b34), muninn 0.1.0 (bcdd9f7), muninn-engine 0.1.0 (7b12b34), muninn-flat 0.1.0 (bcdd9f7), muninn-hnsw 0.1.0 (bcdd9f7), muninn-hnsw-sq8 0.1.0 (bcdd9f7), tantivy tantivy v0.27.0, index_format v7, usearch 2.26.6

## glove-100-angular

1,183,514 vectors, 100 dimensions, metric `cosine`, 10,000 queries, k = 10, 4 threads.

Ground truth check: muninn exact search on 1000 sampled queries matches the provided neighbours for 100.0% of queries by id, 100.0% counting distance ties (recall 1.0000).

### Build

| engine | version | params | build s | threads | index MB | RSS growth MB |
|---|---|---|---|---|---|---|
| muninn-hnsw | 0.1.0 (bcdd9f7) | M=16 efC=200 | 131.8 | 4 | 645.6 | 667.7 |
| muninn-hnsw-sq8 | 0.1.0 (bcdd9f7) | M=16 efC=200 Sq8 rerank=40 | 132.2 | 4 | 764.0 | 667.7 |
| muninn-engine | 0.1.0 (7b12b34) | M=16 efC=200 | 354.4 | 4 | 755.9 | 136.0 |
| hnswlib | 0.8.0 | M=16 efC=200 | 147.4 | 4 | 649.2 | 771.7 |
| faiss-hnsw | 1.16.0 | M=16 efC=200 | 138.4 | 4 | 644.2 | 667.5 |
| usearch | 2.26.6 | M=16 efC=200 | 221.7 | 4 | 649.2 | 803.0 |
| muninn-flat | 0.1.0 (bcdd9f7) | exact | 0.2 | 1 | 473.4 | 473.6 |

### Best QPS at a recall target (one thread / all threads)

Highest throughput among sweep points reaching at least the given recall@k; `ef` in brackets.

| engine | recall ≥ 0.8 | recall ≥ 0.9 | recall ≥ 0.95 | recall ≥ 0.98 | recall ≥ 0.99 |
|---|---|---|---|---|---|
| muninn-hnsw M=16 efC=200 | 5,464 (96) / 13,559 (96) | 1,600 (384) / 4,059 (384) | - | - | - |
| muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40 | 6,139 (96) / 18,929 (96) | 2,057 (384) / 6,155 (384) | - | - | - |
| muninn-engine M=16 efC=200 | 5,332 (96) / 13,469 (96) | 1,528 (384) / 4,066 (384) | - | - | - |
| hnswlib M=16 efC=200 | 5,831 (96) / 14,059 (96) | 1,705 (384) / 4,325 (384) | - | - | - |
| faiss-hnsw M=16 efC=200 | 5,707 (96) / 15,194 (96) | 1,511 (384) / 4,230 (384) | - | - | - |
| usearch M=16 efC=200 | 3,502 (96) / 9,400 (96) | 1,024 (384) / 2,529 (384) | - | - | - |

### muninn-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4675 | 31,989 | 77,390 | 29.3 | 48.9 | 58.8 | yes |
| 16 | 0.5564 | 22,999 | 55,227 | 41.1 | 66.2 | 79.9 | yes |
| 24 | 0.6236 | 16,897 | 40,839 | 56.1 | 88.4 | 107.3 | yes |
| 32 | 0.6692 | 13,700 | 33,150 | 69.7 | 107.0 | 128.0 | yes |
| 48 | 0.7269 | 9,771 | 24,109 | 98.6 | 146.3 | 174.8 | yes |
| 64 | 0.7617 | 7,728 | 19,010 | 126.2 | 180.7 | 213.8 | yes |
| 96 | 0.8050 | 5,464 | 13,559 | 181.0 | 245.1 | 286.1 | yes |
| 128 | 0.8322 | 4,138 | 10,248 | 241.5 | 314.9 | 364.0 | yes |
| 192 | 0.8640 | 2,949 | 7,512 | 341.7 | 430.7 | 482.1 | yes |
| 256 | 0.8850 | 2,282 | 5,811 | 443.4 | 551.1 | 602.2 | yes |
| 384 | 0.9108 | 1,600 | 4,059 | 635.3 | 780.2 | 828.6 | yes |
| 512 | 0.9257 | 1,229 | 3,123 | 828.0 | 1,011.8 | 1,061.7 | yes |

### muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7008 | 13,069 | 37,797 | 73.6 | 107.1 | 125.6 |  |
| 16 | 0.7008 | 13,108 | 37,118 | 73.3 | 106.5 | 126.6 | yes |
| 24 | 0.7008 | 12,942 | 35,928 | 74.2 | 107.8 | 127.7 |  |
| 32 | 0.7008 | 12,069 | 35,280 | 79.5 | 116.7 | 137.4 |  |
| 48 | 0.7260 | 10,378 | 31,514 | 92.8 | 134.0 | 158.0 | yes |
| 64 | 0.7615 | 8,358 | 25,383 | 116.2 | 162.8 | 193.6 | yes |
| 96 | 0.8053 | 6,139 | 18,929 | 160.4 | 215.2 | 253.0 | yes |
| 128 | 0.8321 | 5,056 | 15,562 | 196.5 | 256.9 | 293.3 | yes |
| 192 | 0.8642 | 3,804 | 11,037 | 264.2 | 329.1 | 367.9 | yes |
| 256 | 0.8851 | 2,767 | 8,532 | 364.6 | 451.2 | 500.1 | yes |
| 384 | 0.9108 | 2,057 | 6,155 | 493.1 | 598.8 | 641.0 | yes |
| 512 | 0.9259 | 1,588 | 4,751 | 640.6 | 770.3 | 805.1 | yes |

### muninn-engine M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4700 | 30,088 | 74,592 | 31.3 | 51.0 | 62.4 | yes |
| 16 | 0.5564 | 21,498 | 52,848 | 43.8 | 71.0 | 85.7 | yes |
| 24 | 0.6267 | 16,026 | 39,904 | 59.2 | 93.4 | 110.3 | yes |
| 32 | 0.6706 | 12,880 | 31,878 | 74.0 | 113.8 | 136.9 | yes |
| 48 | 0.7259 | 9,500 | 23,589 | 101.6 | 150.4 | 179.0 | yes |
| 64 | 0.7609 | 7,420 | 18,694 | 131.4 | 186.1 | 220.9 | yes |
| 96 | 0.8051 | 5,332 | 13,469 | 185.1 | 249.5 | 295.4 | yes |
| 128 | 0.8318 | 4,123 | 10,660 | 241.3 | 316.1 | 368.4 | yes |
| 192 | 0.8651 | 2,856 | 7,504 | 351.7 | 446.5 | 497.5 | yes |
| 256 | 0.8853 | 2,224 | 5,826 | 454.5 | 566.7 | 613.9 | yes |
| 384 | 0.9109 | 1,528 | 4,066 | 664.2 | 816.0 | 867.0 | yes |
| 512 | 0.9262 | 1,178 | 3,153 | 865.4 | 1,055.6 | 1,108.8 | yes |

### hnswlib M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4741 | 34,700 | 79,647 | 28.7 | 45.2 | 54.4 | yes |
| 16 | 0.5587 | 24,536 | 56,597 | 40.3 | 63.0 | 76.4 | yes |
| 24 | 0.6270 | 18,183 | 41,682 | 54.4 | 84.3 | 102.8 | yes |
| 32 | 0.6719 | 14,408 | 33,886 | 67.6 | 102.4 | 124.3 | yes |
| 48 | 0.7281 | 10,457 | 24,738 | 94.3 | 137.8 | 164.6 | yes |
| 64 | 0.7620 | 8,243 | 19,736 | 120.2 | 168.4 | 198.4 | yes |
| 96 | 0.8058 | 5,831 | 14,059 | 172.0 | 228.8 | 265.5 | yes |
| 128 | 0.8321 | 4,545 | 11,094 | 223.1 | 288.6 | 327.8 | yes |
| 192 | 0.8635 | 3,159 | 7,900 | 323.4 | 401.0 | 443.3 | yes |
| 256 | 0.8841 | 2,469 | 6,167 | 414.4 | 505.4 | 549.6 | yes |
| 384 | 0.9096 | 1,705 | 4,325 | 601.7 | 723.5 | 766.7 | yes |
| 512 | 0.9259 | 1,355 | 3,305 | 785.7 | 941.6 | 992.7 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 1.4..32.5 us.

### faiss-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4758 | 36,225 | 89,768 | 29.4 | 47.8 | 60.2 | yes |
| 16 | 0.5629 | 26,087 | 64,829 | 40.0 | 64.1 | 77.9 | yes |
| 24 | 0.6313 | 19,067 | 47,522 | 53.4 | 83.8 | 101.0 | yes |
| 32 | 0.6755 | 15,229 | 38,234 | 66.5 | 102.0 | 121.5 | yes |
| 48 | 0.7315 | 10,893 | 27,911 | 91.7 | 135.3 | 161.2 | yes |
| 64 | 0.7639 | 8,613 | 22,363 | 116.4 | 164.4 | 194.6 | yes |
| 96 | 0.8064 | 5,707 | 15,194 | 178.1 | 239.6 | 279.9 | yes |
| 128 | 0.8322 | 4,331 | 11,910 | 232.8 | 302.4 | 348.0 | yes |
| 192 | 0.8628 | 3,232 | 7,905 | 360.9 | 447.9 | 504.5 | yes |
| 256 | 0.8827 | 2,392 | 6,201 | 454.5 | 550.5 | 601.0 | yes |
| 384 | 0.9067 | 1,511 | 4,230 | 678.9 | 805.3 | 852.8 | yes |
| 512 | 0.9212 | 1,142 | 3,201 | 896.1 | 1,053.1 | 1,105.8 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 3.7..51.9 us.

### usearch M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.4717 | 14,688 | 39,392 | 75.4 | 120.3 | 146.7 | yes |
| 16 | 0.5569 | 10,806 | 36,675 | 81.0 | 131.6 | 161.8 | yes |
| 24 | 0.6263 | 9,608 | 19,512 | 141.1 | 218.8 | 257.9 | yes |
| 32 | 0.6716 | 6,961 | 21,851 | 140.7 | 214.6 | 259.3 | yes |
| 48 | 0.7267 | 5,363 | 12,627 | 217.3 | 320.6 | 378.6 | yes |
| 64 | 0.7616 | 3,836 | 12,749 | 273.1 | 395.3 | 468.6 | yes |
| 96 | 0.8046 | 3,502 | 9,400 | 294.2 | 406.0 | 469.9 | yes |
| 128 | 0.8302 | 2,736 | 7,449 | 375.6 | 500.8 | 568.6 | yes |
| 192 | 0.8635 | 1,933 | 5,196 | 545.3 | 712.9 | 802.7 | yes |
| 256 | 0.8842 | 1,457 | 3,889 | 708.2 | 912.0 | 996.9 | yes |
| 384 | 0.9095 | 1,024 | 2,529 | 1,140.9 | 1,499.2 | 1,612.0 | yes |
| 512 | 0.9253 | 683 | 1,750 | 1,693.4 | 2,211.8 | 2,353.4 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was -6.6..200.7 us.

### muninn-flat exact

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| - | 0.9999 | 38 | 119 | 26,245.2 | 27,364.3 | 27,703.0 | yes |

## nytimes-256-angular

290,000 vectors, 256 dimensions, metric `cosine`, 10,000 queries, k = 10, 4 threads.

Ground truth check: muninn exact search on 1000 sampled queries matches the provided neighbours for 92.0% of queries by id, 100.0% counting distance ties (recall 0.9871).

### Build

| engine | version | params | build s | threads | index MB | RSS growth MB |
|---|---|---|---|---|---|---|
| muninn-hnsw | 0.1.0 (bcdd9f7) | M=16 efC=200 | 51.1 | 4 | 339.2 | 343.7 |
| muninn-hnsw-sq8 | 0.1.0 (bcdd9f7) | M=16 efC=200 Sq8 rerank=40 | 51.4 | 4 | 413.4 | 343.7 |
| muninn-engine | 0.1.0 (7b12b34) | M=16 efC=200 | 133.1 | 4 | 444.3 | 121.0 |
| hnswlib | 0.8.0 | M=16 efC=200 | 54.4 | 4 | 340.0 | 373.1 |
| faiss-hnsw | 1.16.0 | M=16 efC=200 | 57.0 | 4 | 338.8 | 350.6 |
| usearch | 2.26.6 | M=16 efC=200 | 86.8 | 4 | 340.0 | 388.8 |
| muninn-flat | 0.1.0 (bcdd9f7) | exact | 0.1 | 1 | 297.0 | 297.1 |

### Best QPS at a recall target (one thread / all threads)

Highest throughput among sweep points reaching at least the given recall@k; `ef` in brackets.

| engine | recall ≥ 0.8 | recall ≥ 0.9 | recall ≥ 0.95 | recall ≥ 0.98 | recall ≥ 0.99 |
|---|---|---|---|---|---|
| muninn-hnsw M=16 efC=200 | 8,035 (48) / 19,225 (48) | 2,015 (256) / 4,995 (256) | - | - | - |
| muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40 | 9,014 (48) / 22,500 (48) | 2,431 (256) / 6,157 (256) | - | - | - |
| muninn-engine M=16 efC=200 | 8,511 (48) / 20,101 (48) | 2,038 (256) / 5,048 (256) | - | - | - |
| hnswlib M=16 efC=200 | 7,611 (48) / 19,399 (48) | 1,908 (256) / 4,557 (256) | - | - | - |
| faiss-hnsw M=16 efC=200 | 8,285 (48) / 18,943 (48) | 1,885 (256) / 4,684 (256) | - | - | - |
| usearch M=16 efC=200 | 4,575 (48) / 9,830 (64) | 1,152 (256) / 3,048 (256) | - | - | - |

### muninn-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.5796 | 26,046 | 62,057 | 36.2 | 60.2 | 72.1 | yes |
| 16 | 0.6705 | 18,766 | 44,374 | 50.6 | 82.8 | 97.9 | yes |
| 24 | 0.7335 | 13,613 | 32,227 | 69.8 | 112.8 | 133.5 | yes |
| 32 | 0.7702 | 11,163 | 25,970 | 85.6 | 135.2 | 161.4 | yes |
| 48 | 0.8141 | 8,035 | 19,225 | 120.0 | 185.6 | 219.7 | yes |
| 64 | 0.8377 | 6,378 | 15,282 | 152.7 | 227.7 | 268.0 | yes |
| 96 | 0.8634 | 4,570 | 11,078 | 216.2 | 303.8 | 354.9 | yes |
| 128 | 0.8778 | 3,573 | 8,848 | 279.7 | 377.0 | 430.6 | yes |
| 192 | 0.8941 | 2,537 | 6,202 | 399.7 | 513.7 | 574.2 | yes |
| 256 | 0.9062 | 2,015 | 4,995 | 507.9 | 633.9 | 688.4 | yes |
| 384 | 0.9210 | 1,403 | 3,456 | 737.1 | 891.0 | 945.0 | yes |
| 512 | 0.9310 | 1,025 | 2,668 | 1,015.3 | 1,205.1 | 1,265.1 | yes |

### muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7956 | 9,854 | 25,597 | 97.5 | 147.8 | 171.3 |  |
| 16 | 0.7956 | 9,902 | 25,564 | 96.9 | 147.0 | 173.4 |  |
| 24 | 0.7956 | 9,909 | 25,658 | 96.9 | 147.0 | 171.4 |  |
| 32 | 0.7956 | 10,202 | 25,721 | 94.1 | 142.8 | 166.7 | yes |
| 48 | 0.8122 | 9,014 | 22,500 | 106.9 | 161.4 | 188.5 | yes |
| 64 | 0.8357 | 7,362 | 18,603 | 131.7 | 193.7 | 225.1 | yes |
| 96 | 0.8621 | 5,470 | 13,498 | 180.4 | 252.5 | 293.6 | yes |
| 128 | 0.8771 | 4,215 | 10,692 | 236.9 | 319.9 | 365.7 | yes |
| 192 | 0.8944 | 3,150 | 7,783 | 322.7 | 411.6 | 458.1 | yes |
| 256 | 0.9064 | 2,431 | 6,157 | 423.1 | 522.0 | 568.6 | yes |
| 384 | 0.9208 | 1,655 | 4,288 | 626.4 | 756.6 | 808.7 | yes |
| 512 | 0.9308 | 1,256 | 3,242 | 828.0 | 990.4 | 1,095.6 | yes |

### muninn-engine M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.5835 | 26,977 | 60,995 | 35.0 | 57.8 | 71.1 | yes |
| 16 | 0.6726 | 18,875 | 44,330 | 50.0 | 82.3 | 97.7 | yes |
| 24 | 0.7379 | 13,949 | 33,501 | 68.0 | 110.2 | 129.4 | yes |
| 32 | 0.7729 | 11,275 | 27,227 | 84.7 | 134.1 | 157.3 | yes |
| 48 | 0.8151 | 8,511 | 20,101 | 113.1 | 173.2 | 203.0 | yes |
| 64 | 0.8385 | 6,741 | 16,086 | 144.2 | 212.8 | 254.3 | yes |
| 96 | 0.8618 | 4,750 | 11,613 | 208.4 | 292.2 | 337.9 | yes |
| 128 | 0.8773 | 3,722 | 9,248 | 269.0 | 361.9 | 414.0 | yes |
| 192 | 0.8932 | 2,602 | 6,492 | 390.6 | 499.4 | 554.0 | yes |
| 256 | 0.9051 | 2,038 | 5,048 | 503.9 | 624.5 | 679.2 | yes |
| 384 | 0.9202 | 1,384 | 3,495 | 747.9 | 901.7 | 955.5 | yes |
| 512 | 0.9304 | 1,041 | 2,662 | 1,001.0 | 1,180.1 | 1,232.1 | yes |

### hnswlib M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.5914 | 27,906 | 60,301 | 35.6 | 58.5 | 69.7 | yes |
| 16 | 0.6762 | 19,881 | 43,717 | 49.4 | 80.2 | 95.5 | yes |
| 24 | 0.7377 | 13,224 | 29,993 | 74.4 | 120.0 | 143.5 | yes |
| 32 | 0.7746 | 10,439 | 23,941 | 93.1 | 149.6 | 177.5 | yes |
| 48 | 0.8154 | 7,611 | 19,399 | 123.8 | 191.5 | 223.1 | yes |
| 64 | 0.8384 | 6,463 | 14,501 | 159.6 | 235.9 | 274.0 | yes |
| 96 | 0.8625 | 4,336 | 10,461 | 233.5 | 326.8 | 376.6 | yes |
| 128 | 0.8777 | 3,403 | 8,323 | 299.3 | 398.6 | 454.3 | yes |
| 192 | 0.8950 | 2,412 | 5,907 | 427.2 | 539.5 | 601.8 | yes |
| 256 | 0.9058 | 1,908 | 4,557 | 544.6 | 669.3 | 731.0 | yes |
| 384 | 0.9214 | 1,309 | 3,196 | 796.5 | 949.7 | 1,007.2 | yes |
| 512 | 0.9312 | 1,018 | 2,475 | 1,096.1 | 1,285.9 | 1,342.0 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was -2.9..64.9 us.

### faiss-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.6081 | 25,647 | 57,984 | 40.9 | 66.5 | 79.3 | yes |
| 16 | 0.6922 | 18,015 | 41,368 | 57.1 | 92.1 | 108.9 | yes |
| 24 | 0.7523 | 13,600 | 31,926 | 72.5 | 115.9 | 136.2 | yes |
| 32 | 0.7840 | 11,117 | 26,054 | 91.5 | 143.2 | 169.0 | yes |
| 48 | 0.8216 | 8,285 | 18,943 | 128.7 | 193.6 | 227.7 | yes |
| 64 | 0.8428 | 6,470 | 15,376 | 157.1 | 228.4 | 269.5 | yes |
| 96 | 0.8665 | 4,472 | 10,263 | 238.7 | 330.6 | 388.0 | yes |
| 128 | 0.8809 | 3,325 | 8,392 | 301.7 | 398.3 | 454.4 | yes |
| 192 | 0.8980 | 2,405 | 6,021 | 429.6 | 537.9 | 594.5 | yes |
| 256 | 0.9085 | 1,885 | 4,684 | 553.9 | 672.0 | 724.9 | yes |
| 384 | 0.9233 | 1,223 | 3,236 | 806.1 | 948.2 | 1,001.7 | yes |
| 512 | 0.9327 | 961 | 2,449 | 1,088.3 | 1,262.2 | 1,320.8 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was -37.5..18.2 us.

### usearch M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.5778 | 14,794 | 38,401 | 73.0 | 118.7 | 140.7 | yes |
| 16 | 0.6709 | 10,587 | 27,454 | 99.6 | 160.7 | 189.7 | yes |
| 24 | 0.7347 | 7,884 | 20,324 | 127.6 | 203.3 | 236.7 | yes |
| 32 | 0.7738 | 6,115 | 12,106 | 169.6 | 273.6 | 325.2 | yes |
| 48 | 0.8165 | 4,575 | 9,231 | 223.8 | 345.7 | 408.8 | yes |
| 64 | 0.8391 | 3,738 | 9,830 | 278.1 | 417.7 | 494.4 | yes |
| 96 | 0.8637 | 2,568 | 6,681 | 421.3 | 602.7 | 694.5 | yes |
| 128 | 0.8783 | 1,992 | 5,296 | 535.9 | 736.5 | 831.3 | yes |
| 192 | 0.8953 | 1,473 | 3,713 | 749.5 | 979.0 | 1,084.4 | yes |
| 256 | 0.9064 | 1,152 | 3,048 | 911.2 | 1,154.8 | 1,254.6 | yes |
| 384 | 0.9215 | 793 | 2,082 | 1,348.4 | 1,641.7 | 1,741.6 | yes |
| 512 | 0.9315 | 597 | 1,566 | 1,854.0 | 2,263.9 | 2,399.3 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 7.1..97.8 us.

### muninn-flat exact

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| - | 0.9848 | 78 | 253 | 12,769.3 | 13,076.0 | 13,223.4 | yes |

## sift-128-euclidean

1,000,000 vectors, 128 dimensions, metric `l2`, 10,000 queries, k = 10, 4 threads.

Ground truth check: muninn exact search on 1000 sampled queries matches the provided neighbours for 99.2% of queries by id, 100.0% counting distance ties (recall 0.9992).

### Build

| engine | version | params | build s | threads | index MB | RSS growth MB |
|---|---|---|---|---|---|---|
| muninn-hnsw | 0.1.0 (bcdd9f7) | M=16 efC=200 | 96.9 | 4 | 657.5 | 676.4 |
| muninn-hnsw-sq8 | 0.1.0 (bcdd9f7) | M=16 efC=200 Sq8 rerank=40 | 97.3 | 4 | 789.5 | 676.4 |
| muninn-engine | 0.1.0 (7b12b34) | M=16 efC=200 | 256.5 | 4 | 787.8 | 152.7 |
| hnswlib | 0.8.0 | M=16 efC=200 | 111.0 | 4 | 660.6 | 768.4 |
| faiss-hnsw | 1.16.0 | M=16 efC=200 | 111.4 | 4 | 656.3 | 673.9 |
| usearch | 2.26.6 | M=16 efC=200 | 157.8 | 4 | 660.6 | 762.7 |
| muninn-flat | 0.1.0 (bcdd9f7) | exact | 0.2 | 1 | 512.0 | 512.2 |

### Best QPS at a recall target (one thread / all threads)

Highest throughput among sweep points reaching at least the given recall@k; `ef` in brackets.

| engine | recall ≥ 0.8 | recall ≥ 0.9 | recall ≥ 0.95 | recall ≥ 0.98 | recall ≥ 0.99 |
|---|---|---|---|---|---|
| muninn-hnsw M=16 efC=200 | 27,574 (16) / 66,741 (16) | 13,982 (32) / 33,666 (32) | 9,326 (64) / 22,670 (64) | 6,184 (96) / 14,545 (96) | 3,311 (192) / 8,615 (192) |
| muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40 | 13,621 (10) / 38,550 (10) | 13,621 (10) / 38,550 (10) | 9,426 (64) / 26,148 (64) | 6,897 (96) / 18,958 (96) | 3,860 (192) / 10,291 (192) |
| muninn-engine M=16 efC=200 | 20,182 (24) / 48,515 (24) | 16,473 (32) / 39,099 (32) | 9,310 (64) / 22,719 (64) | 6,545 (96) / 16,164 (96) | 3,544 (192) / 8,681 (192) |
| hnswlib M=16 efC=200 | 25,449 (16) / 56,875 (16) | 15,526 (32) / 35,758 (32) | 8,883 (64) / 19,997 (64) | 5,953 (96) / 14,854 (96) | 3,352 (192) / 8,192 (192) |
| faiss-hnsw M=16 efC=200 | 25,306 (16) / 61,027 (16) | 16,132 (32) / 37,407 (32) | 8,920 (64) / 21,238 (64) | 5,893 (96) / 15,399 (96) | 4,562 (128) / 11,786 (128) |
| usearch M=16 efC=200 | 14,075 (16) / 36,798 (16) | 8,542 (32) / 22,459 (32) | 4,697 (64) / 12,483 (64) | 3,424 (96) / 8,877 (96) | 1,938 (192) / 5,120 (192) |

### muninn-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7073 | 38,021 | 94,907 | 25.4 | 37.8 | 44.9 | yes |
| 16 | 0.8008 | 27,574 | 66,741 | 35.5 | 49.7 | 59.2 | yes |
| 24 | 0.8654 | 20,081 | 47,469 | 49.6 | 65.9 | 75.6 | yes |
| 32 | 0.9031 | 13,982 | 33,666 | 72.1 | 90.6 | 102.5 | yes |
| 48 | 0.9434 | 10,008 | 24,203 | 101.9 | 123.7 | 135.8 | yes |
| 64 | 0.9634 | 9,326 | 22,670 | 109.5 | 132.8 | 143.7 | yes |
| 96 | 0.9815 | 6,184 | 14,545 | 166.4 | 199.0 | 210.8 | yes |
| 128 | 0.9891 | 4,574 | 11,380 | 226.3 | 267.3 | 283.4 | yes |
| 192 | 0.9950 | 3,311 | 8,615 | 311.2 | 381.5 | 406.0 | yes |
| 256 | 0.9973 | 2,757 | 7,046 | 374.6 | 453.3 | 477.5 | yes |
| 384 | 0.9988 | 1,944 | 4,997 | 531.3 | 647.9 | 680.5 | yes |
| 512 | 0.9992 | 1,475 | 3,847 | 699.2 | 858.3 | 907.0 | yes |

### muninn-hnsw-sq8 M=16 efC=200 Sq8 rerank=40

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.9267 | 13,621 | 38,550 | 74.3 | 90.1 | 100.1 | yes |
| 16 | 0.9267 | 13,442 | 38,041 | 75.2 | 91.2 | 101.1 |  |
| 24 | 0.9267 | 13,427 | 37,740 | 75.3 | 91.8 | 102.4 |  |
| 32 | 0.9267 | 13,475 | 38,102 | 75.1 | 91.0 | 101.6 |  |
| 48 | 0.9436 | 11,461 | 32,885 | 88.8 | 106.2 | 116.2 | yes |
| 64 | 0.9637 | 9,426 | 26,148 | 108.6 | 127.7 | 137.5 | yes |
| 96 | 0.9815 | 6,897 | 18,958 | 149.6 | 173.7 | 183.3 | yes |
| 128 | 0.9893 | 5,395 | 14,906 | 191.8 | 222.5 | 234.6 | yes |
| 192 | 0.9950 | 3,860 | 10,291 | 268.9 | 313.7 | 327.6 | yes |
| 256 | 0.9973 | 2,803 | 8,155 | 369.0 | 436.1 | 457.2 | yes |
| 384 | 0.9988 | 2,110 | 5,982 | 491.3 | 585.9 | 609.9 | yes |
| 512 | 0.9992 | 1,649 | 4,641 | 627.5 | 755.0 | 791.2 | yes |

### muninn-engine M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7076 | 36,109 | 85,544 | 26.8 | 39.3 | 46.6 | yes |
| 16 | 0.7995 | 26,822 | 64,476 | 36.5 | 51.2 | 59.6 | yes |
| 24 | 0.8661 | 20,182 | 48,515 | 49.1 | 65.4 | 75.6 | yes |
| 32 | 0.9046 | 16,473 | 39,099 | 60.6 | 78.7 | 90.2 | yes |
| 48 | 0.9447 | 11,832 | 28,378 | 85.6 | 105.9 | 116.5 | yes |
| 64 | 0.9644 | 9,310 | 22,719 | 109.5 | 133.6 | 144.4 | yes |
| 96 | 0.9820 | 6,545 | 16,164 | 156.9 | 188.4 | 201.5 | yes |
| 128 | 0.9894 | 5,084 | 12,572 | 202.3 | 242.6 | 258.3 | yes |
| 192 | 0.9951 | 3,544 | 8,681 | 290.5 | 348.7 | 368.8 | yes |
| 256 | 0.9971 | 2,734 | 6,927 | 377.5 | 456.4 | 481.0 | yes |
| 384 | 0.9987 | 1,912 | 4,797 | 539.7 | 656.2 | 693.0 | yes |
| 512 | 0.9991 | 1,483 | 3,807 | 694.2 | 853.1 | 900.2 | yes |

### hnswlib M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7095 | 32,316 | 74,285 | 30.4 | 44.0 | 52.4 | yes |
| 16 | 0.8016 | 25,449 | 56,875 | 39.4 | 54.3 | 64.8 | yes |
| 24 | 0.8660 | 19,465 | 42,928 | 53.2 | 70.3 | 80.6 | yes |
| 32 | 0.9028 | 15,526 | 35,758 | 66.8 | 85.2 | 96.1 | yes |
| 48 | 0.9433 | 11,240 | 25,262 | 95.9 | 118.3 | 133.6 | yes |
| 64 | 0.9635 | 8,883 | 19,997 | 122.7 | 149.9 | 164.6 | yes |
| 96 | 0.9816 | 5,953 | 14,854 | 177.6 | 212.5 | 226.8 | yes |
| 128 | 0.9892 | 4,583 | 11,028 | 227.6 | 272.0 | 288.1 | yes |
| 192 | 0.9952 | 3,352 | 8,192 | 310.6 | 372.9 | 395.1 | yes |
| 256 | 0.9974 | 2,580 | 6,389 | 405.7 | 487.3 | 513.2 | yes |
| 384 | 0.9986 | 1,831 | 4,342 | 569.9 | 691.7 | 727.8 | yes |
| 512 | 0.9990 | 1,375 | 3,529 | 755.4 | 919.6 | 965.9 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 0.3..7.7 us.

### faiss-hnsw M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7157 | 33,654 | 83,197 | 32.4 | 48.1 | 59.9 | yes |
| 16 | 0.8075 | 25,306 | 61,027 | 42.5 | 58.7 | 71.3 | yes |
| 24 | 0.8735 | 18,651 | 46,357 | 57.0 | 76.0 | 87.7 | yes |
| 32 | 0.9108 | 16,132 | 37,407 | 70.7 | 90.8 | 101.9 | yes |
| 48 | 0.9485 | 11,075 | 28,527 | 93.4 | 116.8 | 129.6 | yes |
| 64 | 0.9675 | 8,920 | 21,238 | 121.7 | 146.7 | 160.0 | yes |
| 96 | 0.9835 | 5,893 | 15,399 | 178.5 | 212.5 | 227.0 | yes |
| 128 | 0.9904 | 4,562 | 11,786 | 244.2 | 286.0 | 302.4 | yes |
| 192 | 0.9955 | 3,076 | 8,720 | 333.3 | 392.4 | 411.0 | yes |
| 256 | 0.9974 | 2,478 | 6,598 | 441.6 | 518.8 | 544.1 | yes |
| 384 | 0.9987 | 1,630 | 4,660 | 622.0 | 731.2 | 767.2 | yes |
| 512 | 0.9990 | 1,300 | 3,609 | 804.9 | 948.3 | 993.9 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was -8.6..25.6 us.

### usearch M=16 efC=200

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| 10 | 0.7088 | 18,891 | 49,962 | 59.0 | 84.3 | 101.1 | yes |
| 16 | 0.8005 | 14,075 | 36,798 | 77.5 | 107.7 | 125.3 | yes |
| 24 | 0.8660 | 10,661 | 27,924 | 101.8 | 134.5 | 154.3 | yes |
| 32 | 0.9031 | 8,542 | 22,459 | 126.5 | 162.7 | 180.9 | yes |
| 48 | 0.9428 | 6,296 | 16,707 | 167.4 | 207.6 | 228.0 | yes |
| 64 | 0.9629 | 4,697 | 12,483 | 227.7 | 277.9 | 300.0 | yes |
| 96 | 0.9811 | 3,424 | 8,877 | 322.1 | 397.7 | 426.5 | yes |
| 128 | 0.9889 | 2,822 | 7,126 | 382.4 | 467.5 | 495.5 | yes |
| 192 | 0.9949 | 1,938 | 5,120 | 563.4 | 693.1 | 734.1 | yes |
| 256 | 0.9972 | 1,542 | 3,940 | 702.8 | 871.3 | 920.0 | yes |
| 384 | 0.9987 | 1,088 | 2,764 | 991.4 | 1,237.4 | 1,312.7 | yes |
| 512 | 0.9991 | 839 | 2,142 | 1,266.0 | 1,598.4 | 1,697.9 | yes |

Latency columns include the Python call; per-query overhead versus the batch call was 5.6..37.1 us.

### muninn-flat exact

| ef | recall@k | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | Pareto |
|---|---|---|---|---|---|---|---|
| - | 0.9994 | 42 | 132 | 24,001.0 | 24,587.7 | 24,878.2 | yes |

## MS MARCO passages, BM25 top-k

### Analyzer `default`

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 4 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn | 0.1.0 (bcdd9f7) | 1 | 6.0 | 167,143 | 83.2 | 794 | 2,077 | 834.5 | 3,731.8 | 7,162.9 | 10.00 |
| tantivy | tantivy v0.27.0, index_format v7 | 1 | 6.3 | 157,608 | 74.1 | 1,341 | 3,766 | 430.9 | 2,423.1 | 4,591.4 | 10.00 |

- muninn: 5.5 s indexing + 0.5 s writing; analyzer: alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped
- tantivy indexing with 4 threads: 3.9 s (255,058 docs/s; 2.6 s indexing + 1.3 s merging), 74.8 MB
- tantivy: 6.3 s indexing + 0.0 s merging; analyzer: SimpleTokenizer, LowerCaser, RemoveLongFilter(256) (tokens of at most 255 bytes kept)
- Result agreement muninn vs tantivy: mean overlap@10 0.9997, identical top-10 sets 99.8%, same top-1 99.9%.

### Analyzer `default`, end to end through the engine

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 4 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn-engine | 0.1.0 (7b12b34) | 1 | 15.2 | 65,886 | 520.1 | 785 | 2,060 | 854.0 | 3,757.4 | 7,156.8 | 10.00 |
| muninn | 0.1.0 (7b12b34) | 1 | 6.1 | 164,437 | 83.2 | 788 | 2,073 | 842.1 | 3,745.9 | 7,145.1 | 10.00 |

- muninn-engine index size is the whole live collection, including the stored passage text, id map and write-ahead log.
- muninn-engine: 9.7 s indexing + 5.4 s flush + compaction; analyzer: alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped
- muninn: 5.6 s indexing + 0.5 s writing; analyzer: alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped
- Result agreement muninn-engine vs muninn: mean overlap@10 1.0000, identical top-10 sets 100.0%, same top-1 100.0%.

### Analyzer `english`

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 4 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn | 0.1.0 (bcdd9f7) | 1 | 7.2 | 138,741 | 67.2 | 1,832 | 5,175 | 317.1 | 1,813.7 | 3,172.0 | 10.00 |
| tantivy | tantivy v0.27.0, index_format v7 | 1 | 14.5 | 69,129 | 60.2 | 1,952 | 5,447 | 262.4 | 1,827.9 | 3,425.7 | 10.00 |

- muninn: 6.8 s indexing + 0.4 s writing; analyzer: alphanumeric runs, Unicode lowercase, Latin diacritic folding, tokens > 255 bytes dropped, Lucene English stop words, Porter stemmer
- tantivy indexing with 4 threads: 6.9 s (144,647 docs/s; 5.9 s indexing + 1.0 s merging), 61.0 MB
- tantivy: 14.5 s indexing + 0.0 s merging; analyzer: SimpleTokenizer, LowerCaser, AsciiFoldingFilter, RemoveLongFilter(256), StopWordFilter(English, Lucene list), Stemmer(English, Snowball)
- Result agreement muninn vs tantivy: mean overlap@10 0.9798, identical top-10 sets 91.5%, same top-1 97.7%.

### Analyzer `english`, end to end through the engine

1,000,000 passages (335 MB of text), 6,980 queries (queries.dev.small.tsv), k = 10, 4 query threads.

| engine | version | index threads | index s | docs/s | index MB | QPS 1t | QPS 4t | p50 us | p95 us | p99 us | hits/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muninn-engine | 0.1.0 (7b12b34) | 1 | 15.6 | 64,184 | 510.8 | 1,799 | 4,978 | 326.7 | 1,819.1 | 3,218.9 | 10.00 |
| muninn | 0.1.0 (7b12b34) | 1 | 7.5 | 133,987 | 67.2 | 1,812 | 5,037 | 321.9 | 1,819.3 | 3,194.3 | 10.00 |

- muninn-engine index size is the whole live collection, including the stored passage text, id map and write-ahead log.
- muninn-engine: 11.4 s indexing + 4.1 s flush + compaction; analyzer: alphanumeric runs, Unicode lowercase, Latin diacritic folding, tokens > 255 bytes dropped, Lucene English stop words, Porter stemmer
- muninn: 7.0 s indexing + 0.4 s writing; analyzer: alphanumeric runs, Unicode lowercase, Latin diacritic folding, tokens > 255 bytes dropped, Lucene English stop words, Porter stemmer
- Result agreement muninn-engine vs muninn: mean overlap@10 1.0000, identical top-10 sets 100.0%, same top-1 100.0%.

