# Khai báo thư viện để sử dụng các hàm thống
# kê liên quan đến phân phối Chuẩn và Nhị thức

'''
Xác suất xuất hiện mặt ngửa của một đồng xu không cân đối là 0.45
Tung đồng xu 200 lần.
Tính xác suất
    a) có đúng 150 lần xuất hiện mặt ngửa.
    b) Có từ 30 đến 45 lần xuất hiện mặt ngửa
'''
# Gọi X là số lần xuất hiện mặt ngửa, thì X ~ B(200;0.45)
# Let's go...

from scipy.stats import binom, norm

# Tính câu a)

B = binom(200,0.45)
f= B.pmf(150)
# Có thể viết gộp thành 1 dòng:
f2 = binom.pmf(150,200,0.45)

print('a)',f)
print('or a)',f2)


# Tính chính xác câu b)

S = 0
for i in range(30,46):
  S = S + B.pmf(i)

print('b)',S)

# Bây giờ xấp xỉ câu b)
# X ~ N(90;49.5)


# Giả sử Mean = 20  và Standard Deviation = 2.64575
g = norm.cdf(45,90,49.5**0.5) - norm.cdf(30,90,49.5**0.5)

print(g)