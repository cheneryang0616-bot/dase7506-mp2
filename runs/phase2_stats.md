# MP2 Phase-2 development profile

## Split: train  (n=4755)

- Mean correct candidates per question: **3.0185**
- Correct-count distribution (0..4): {'1': 711, '2': 732, '3': 1070, '4': 2242}
- Per-position accuracy %: [75.37, 75.04, 75.79, 75.65]
- Parseable candidates per question: {'1': 27, '2': 81, '3': 628, '4': 4019}
- Distinct valid answers per question: {'1': 2641, '2': 1338, '3': 608, '4': 168}
- Hard subset (>=2 distinct valid answers): {'n': 2114, 'pct': 44.46}
- Majority vote: correct=4054 wrong=701 tie=545 all-fail=0 -> **85.26%**
- Surface means (correct vs wrong): {'correct': {'n': 14353, 'chars': 464.84, 'lines': 24.4, 'digits': 18.4, 'pct_boxed': 81.0}, 'wrong': {'n': 4667, 'chars': 527.41, 'lines': 25.31, 'digits': 20.66, 'pct_boxed': 65.0}}
- Length buckets: {'0-200': {'correct': 470, 'total': 634, 'pct_correct': 74.13}, '200-400': {'correct': 5653, 'total': 7176, 'pct_correct': 78.78}, '400-700': {'correct': 6445, 'total': 8384, 'pct_correct': 76.87}, '700-1000000000': {'correct': 1785, 'total': 2826, 'pct_correct': 63.16}}
- Boxed marker vs correctness: {'boxed': {'correct': 11654, 'total': 14675, 'pct_correct': 79.41}, 'not_boxed': {'correct': 2699, 'total': 4345, 'pct_correct': 62.12}}

Top vote patterns:
  - `4`: 2242 (47.15%)
  - `3+1`: 885 (18.61%)
  - `2+1+1`: 496 (10.43%)
  - `3 (+1fail)`: 328 (6.9%)
  - `2+2`: 228 (4.79%)
  - `2+1 (+1fail)`: 188 (3.95%)
  - `1+1+1+1`: 168 (3.53%)
  - `1+1+1 (+1fail)`: 112 (2.36%)
  - `2 (+2fail)`: 44 (0.93%)
  - `1+1 (+2fail)`: 37 (0.78%)
  - `1 (+3fail)`: 27 (0.57%)

## Split: validation  (n=798)

- Mean correct candidates per question: **3.0276**
- Correct-count distribution (0..4): {'1': 117, '2': 123, '3': 179, '4': 379}
- Per-position accuracy %: [78.2, 75.31, 74.56, 74.69]
- Parseable candidates per question: {'1': 3, '2': 25, '3': 98, '4': 672}
- Distinct valid answers per question: {'1': 443, '2': 227, '3': 108, '4': 20}
- Hard subset (>=2 distinct valid answers): {'n': 355, 'pct': 44.49}
- Majority vote: correct=688 wrong=110 tie=88 all-fail=0 -> **86.22%**
- Surface means (correct vs wrong): {'correct': {'n': 2416, 'chars': 456.42, 'lines': 23.72, 'digits': 18.21, 'pct_boxed': 80.0}, 'wrong': {'n': 776, 'chars': 548.33, 'lines': 26.39, 'digits': 20.68, 'pct_boxed': 63.0}}
- Length buckets: {'0-200': {'correct': 91, 'total': 115, 'pct_correct': 79.13}, '200-400': {'correct': 963, 'total': 1180, 'pct_correct': 81.61}, '400-700': {'correct': 1095, 'total': 1447, 'pct_correct': 75.67}, '700-1000000000': {'correct': 267, 'total': 450, 'pct_correct': 59.33}}
- Boxed marker vs correctness: {'boxed': {'correct': 1921, 'total': 2409, 'pct_correct': 79.74}, 'not_boxed': {'correct': 495, 'total': 783, 'pct_correct': 63.22}}

Top vote patterns:
  - `4`: 379 (47.49%)
  - `3+1`: 151 (18.92%)
  - `2+1+1`: 88 (11.03%)
  - `3 (+1fail)`: 50 (6.27%)
  - `2+2`: 34 (4.26%)
  - `2+1 (+1fail)`: 28 (3.51%)
  - `1+1+1+1`: 20 (2.51%)
  - `1+1+1 (+1fail)`: 20 (2.51%)
  - `1+1 (+2fail)`: 14 (1.75%)
  - `2 (+2fail)`: 11 (1.38%)
  - `1 (+3fail)`: 3 (0.38%)
