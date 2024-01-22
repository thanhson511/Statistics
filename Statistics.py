# Khai báo thư viện để sử dụng các hàm thống
# kê liên quan đến phân phối Poisson và Nhị thức

from scipy.stats import poisson, binom

# Tạo ngẫu nhiên mẫu có phân phối Poisson, Nhị thức
ps = poisson(0.75).rvs(80)
bn = binom(10,0.7).rvs(50)

print("Poisson:",ps)
print("Binomial",bn)

print(type(bn))
print(type(ps))


for i in range(10):
  print(f'Xác suất để X = {i} là ',binom(8,0.6).pmf(i))

for i in range(10):
  print(f'Xác suất để X = {i} là ',poisson(0.75).pmf(i))