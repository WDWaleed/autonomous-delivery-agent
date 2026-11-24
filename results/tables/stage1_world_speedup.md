**Runtime Comparison and Speedup Factor: Pure-Python Loop vs NumPy Vectorized World Construction**

*Group Stamp: Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)*

| Grid Size (N) | Total Cells | Loop Mean (ms) | Loop 95% CI (ms) | Vectorized Mean (ms) | Vec 95% CI (ms) | Speedup (x) |
| ------------- | ----------- | -------------- | ---------------- | -------------------- | --------------- | ----------- |
| 20            | 400         | 0.737          | 0.177            | 0.39                 | 0.065           | 1.89        |
| 22            | 484         | 4.646          | 7.843            | 0.541                | 0.075           | 8.59        |
| 30            | 900         | 2.01           | 0.628            | 0.805                | 0.113           | 2.5         |
