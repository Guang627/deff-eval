129 submissions (resolve rate 0.1-0.9), 40 tasks per benchmark, 300 draws per submission, 500 bootstrap resamples. Averages over submissions; pp except coverage. Median repository share of variance: 0.034.

| tier    | scheme       |   sd_mean |   bias_mean |   rmse_mean |   cov_mean |   cov_repo_target_mean |   width_mean |
|:--------|:-------------|----------:|------------:|------------:|-----------:|-----------------------:|-------------:|
| natural | iid          |      7.09 |        0.06 |        7.09 |      0.948 |                  0.904 |        28.18 |
| natural | stratified   |      5.28 |        0.02 |        5.28 |      0.922 |                  0.836 |        19.94 |
| natural | repo_cluster |      7.09 |        0.06 |        7.09 |      0.912 |                  0.884 |        30.97 |
| natural | wilson       |      7.09 |        0.06 |        7.09 |      0.96  |                  0.926 |        27.57 |
| 8x5     | iid          |      6.81 |       -3.09 |        7.87 |      0.928 |                  0.938 |        28.54 |
| 8x5     | stratified   |      5.09 |       -0.08 |        5.8  |      0.906 |                  0.881 |        20.5  |
| 8x5     | repo_cluster |      6.81 |       -3.09 |        7.87 |      0.907 |                  0.913 |        29.26 |
| 8x5     | wilson       |      6.81 |       -3.09 |        7.87 |      0.931 |                  0.953 |        27.88 |
| 4x10    | iid          |      7.9  |       -3.19 |        8.73 |      0.892 |                  0.892 |        28.34 |
| 4x10    | stratified   |      5.34 |       -0.3  |        5.7  |      0.904 |                  0.844 |        19.98 |
| 4x10    | repo_cluster |      7.9  |       -3.19 |        8.73 |      0.821 |                  0.811 |        29.02 |
| 4x10    | wilson       |      7.9  |       -3.19 |        8.73 |      0.9   |                  0.913 |        27.7  |
| 2x20    | iid          |      8.99 |       -1.34 |        9.3  |      0.861 |                  0.831 |        27.92 |
| 2x20    | stratified   |      5.56 |       -0.77 |        5.93 |      0.887 |                  0.814 |        19.8  |
| 2x20    | repo_cluster |      8.99 |       -1.34 |        9.3  |      0.528 |                  0.508 |        16.59 |
| 2x20    | wilson       |      8.99 |       -1.34 |        9.3  |      0.881 |                  0.86  |        27.35 |
| 1x40    | iid          |      8.81 |       -3.97 |        9.82 |      0.831 |                  0.797 |        27.5  |
| 1x40    | stratified   |      5.63 |       -1.01 |        6.1  |      0.896 |                  0.781 |        20.16 |
| 1x40    | repo_cluster |      8.81 |       -3.97 |        9.82 |      0.004 |                  0     |         0    |
| 1x40    | wilson       |      8.81 |       -3.97 |        9.82 |      0.855 |                  0.828 |        27.06 |
