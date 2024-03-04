import matplotlib.pyplot as plt 
import numpy as np 
x=np.array([151,174,138,186,128,136,179,163,152,131])
y=np.array([68,81,56,91,47,57,76,72,62,48])

def linear_regression(x,y):
    n=len(x)
    if (len(y)!=n):
        raise ValueError("x is greater than y")
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    ss_xy=n*np.sum(x*y)-np.sum(x)*np.sum(y)
    ss_xx=n*np.sum(x*x)-pow(np.sum(x)^2)
    slope=ss_xy/ss_xx
    intercept=y_mean-slope*x_mean
    return{"slope":slope,"intercept":intercept}
coefficients=linear_regression(x,y)
def predict(x,coefficients):
    return coefficients["slope"]*x+coefficients["intercept"]
plt.scatter(x,y,marker="o", color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression LSM')
plt.plot(x,predict(x,coefficients),color="red")
plt.show()
new_x=np.array([10])
predicted_y=predict(new_x,coefficients)
print("Predicted y value for new x",predicted_y)