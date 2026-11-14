# Evaluation

- policy: random-legal (analytic expectation)
- data: data/tensors/official-v1/test.h5

| Group | N | NLL | Top-1 % | Top-3 % | Top-5 % | Mean rank | ECE % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 75000 | 3.2248 | 6.21 | 16.25 | 24.23 | 15.28 | - |
| opening | 25000 | 3.5277 | 3.37 | 9.75 | 15.73 | 18.19 | - |
| middlegame | 25000 | 3.4280 | 5.15 | 13.20 | 19.37 | 17.83 | - |
| endgame | 25000 | 2.7187 | 10.12 | 25.81 | 37.59 | 9.81 | - |
