### Experiment results (test split, n=150 tickets)

| metric | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| routing accuracy (4 routing nodes) | 0.997 | 0.834 | 0.950 | 0.950 | 0.997 |
|   scored routing decisions (n) | 297 | 290 | 301 | 301 | 306 |
| decision accuracy incl. guard | 0.989 | 0.875 | 0.965 | 0.965 | 0.998 |
| full-path exact match | 0.967 | 0.600 | 0.833 | 0.833 | 0.933 |
| final-action accuracy | 0.973 | 0.627 | 0.833 | 0.833 | 0.933 |
| guard recall (adversarial) | 1.000 | 0.455 | 1.000 | 1.000 | 1.000 |
| guard FPR (clean) | 0.029 | 0.007 | 0.007 | 0.007 | 0.000 |
| routing latency p50 ms | 2380 | 85 | 169 | 3694 | 97 |
| routing latency p95 ms | 2910 | 118 | 2940 | 7722 | 116 |
| end-to-end p50 s | 15.7 | 4.4 | 9.6 | 17.5 | 5.4 |
| end-to-end p95 s | 24.3 | 11.5 | 20.7 | 32.3 | 17.1 |
| LLM calls / ticket | 4.69 | 0.78 | 2.57 | 2.60 | 1.31 |
|   of which routing | 3.63 | 0.00 | 1.55 | 1.57 | 0.16 |
| LLM tokens / ticket | 1711 | 431 | 1112 | 1122 | 753 |
|   routing tokens | 1080 | 0 | 482 | 492 | 51 |
|   in-node tokens | 631 | 431 | 630 | 630 | 703 |
| fallback rate (of Laya decisions) | 0.000 | 0.000 | 0.413 | 0.420 | 0.042 |
| final_review first-pass rate | 1.000 | 0.929 | 0.908 | 0.908 | 0.911 |
| peak VRAM (MiB, whole GPU) | 5797 | 6842 | 6842 | 6842 | 6862 |
| engine errors | 0 | 0 | 0 | 0 | 0 |

Routing latency by router (ms, p50 / p95 / n):

- E1: llm 2380 / 2910 / 298
- E2: laya 85 / 118 / 309
- E3: laya 80 / 146 / 155; llm_fallback 2529 / 3147 / 150
- E4: laya 1855 / 5516 / 154; llm_fallback 4212 / 8518 / 151
- E5: laya 97 / 116 / 305; llm_fallback 2689 / 2689 / 1

Routing accuracy by node:

| node | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| classify | 0.993 | 0.841 | 0.957 | 0.957 | 0.993 |
| lookup_order | 1.000 | 0.884 | 0.944 | 0.944 | 1.000 |
| lookup_order_delivery | 1.000 | 0.786 | 0.848 | 0.848 | 1.000 |
| policy_check | 1.000 | 0.782 | 1.000 | 1.000 | 1.000 |

Final-action accuracy by slice:

| slice | n | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|---|
| all | 150 | 0.973 | 0.627 | 0.833 | 0.833 | 0.933 |
| hard | 33 | 0.909 | 0.424 | 0.697 | 0.697 | 0.909 |
| adversarial | 11 | 1.000 | 0.727 | 1.000 | 1.000 | 1.000 |
| long | 4 | 0.750 | 0.250 | 1.000 | 1.000 | 1.000 |
| two_issues | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cat=adversarial | 11 | 1.000 | 0.727 | 1.000 | 1.000 | 1.000 |
| cat=delivery | 34 | 0.971 | 0.647 | 0.824 | 0.824 | 1.000 |
| cat=other | 12 | 0.917 | 1.000 | 0.917 | 0.917 | 0.917 |
| cat=refund | 72 | 0.972 | 0.625 | 0.806 | 0.806 | 0.875 |
| cat=technical | 21 | 1.000 | 0.333 | 0.810 | 0.810 | 1.000 |

![experiments](reports/experiments_test.png)

