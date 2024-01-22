# OK
# Khai báo thư viện để sử dụng các hàm thống
# kê liên quan đến phân phối Chuẩn và Nhị thức

'''
Xác suất nảy mầm của mỗi hạt giống là 0.8
Gieo thử 500 hạt.
Tính xác suất
    a) có đúng 375 hạt nảy mầm.
    b) Có từ 390 đến 450 hạt nảy mầm.
'''
# Gọi X là số lần xuất hiện mặt ngửa, thì X ~ B(500;0.8)
# Let's go...

from scipy.stats import binom, norm


B = binom(500,0.8)
f= B.pmf(375)
# Có thể viết gộp thành 1 dòng:
# B = binom.pmf(25,200,0.45)

print('a)',f)

S = 0
for i in range(390,451):
  S = S + B.pmf(i)

print('b)',S)

# Bây giờ xấp xỉ X ~ N(400;80)

g = norm.cdf(5.59)-norm.cdf(-1.12)

print(g)