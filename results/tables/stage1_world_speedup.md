**Runtime Comparison and Speedup Factor: Pure-Python Loop vs NumPy Vectorized World Construction**

*Group Stamp: Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)*

| Grid Size (N) | Total Cells | Loop Mean (ms) | Loop 95% CI (ms) | Vectorized Mean (ms) | Vec 95% CI (ms) | Speedup (x) |
| ------------- | ----------- | -------------- | ---------------- | -------------------- | --------------- | ----------- |
| 20            | 400         | 1.626          | 2.003            | 0.308                | 0.024           | 5.28        |
| 22            | 484         | 0.708          | 0.096            | 0.308                | 0.034           | 2.3         |
| 30            | 900         | 1.214          | 0.119            | 0.381                | 0.02            | 3.18        |
| 200           | 40000       | 88.341         | 8.479            | 11.592               | 0.867           | 7.62        |
| 1000          | 1000000     | 3701.035       | 1446.079         | 372.06               | 68.276          | 9.95        |
