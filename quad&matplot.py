import numpy as np
import matplotlib.pyplot as plt
#画图,特别重要，必须要掌握！！！！！！

from scipy.integrate import quad, dblquad
#
# print(np.exp(2))
# exit()

# def qiujifen():
#      var, err = quad(lambda x:np.exp(x)*2*np.sin(x)*np.cos(x),
#                      1.5,#x下届
#                      np.pi/2)#x的上届
#      print('积分结果为:', var)


#我从几到几，按距离取多少个点
#拿横坐标
x = np.linspace(-np.pi, np.pi, 200)

sin_x = np.sin(x)
cos_x = np.cos(x)
another = np.sin(2*x)*np.cos(x)
#开始用matplot画图
plt.plot(x, sin_x)
plt.plot(x, cos_x)
plt.plot(x, another)
plt.legend(['sinx', 'cosx','sin2x*cosx'])
plt.title('COSx&Sinx')
plt.show()


