from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

# Finding line of best fit y = mx + b 
# m = mean (x) * mean (y) - mean (xy) / (mean x) squared - mean (x squared)
# b = mean(y) - m * mean(x)

# xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

def create_dataset(how_many, variance, step=2, correlation=False):
# how_many datapoints to create
# how variable dataset to be
# how far average to step up y-value per point 
# correlation true, step pos. correlation false, step neg
    val = 1
    ys = []
    for i in range (how_many):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs)**2 - mean(xs**2)) )
    b = mean(ys) - m * mean(xs)
    return m, b # prints 0.42857142871, 4.0

def squared_error(ys_orig, ys_line):
    return sum( (ys_line - ys_orig) **2 ) 

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regression = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return (1 - (squared_error_regression / squared_error_y_mean) )
    # determine how accurate our model using r-squared / coefficent of determination
# squared error is the distance between the point and best_fit_line 
# value is squared to ensure positive values 
# penalizes for outliers instead of absolute values
# r^2 = 1 - ( (SE) * y-hat / (SE) * mean(y) )
# comparing accuracy of average y-line to best_fit_line
# when comaprison ratio is low, accuracy high
# when r^2 is high, accuracy is high 

# xs, ys = create_dataset(40, 40, 2, correlation = 'pos') # prints 0.60133246
# variance decrease, r-squared increase
# xs, ys = create_dataset(40, 10, 2, correlation = False)  # prints 0.000795
xs, ys = create_dataset(40, 10, 2, correlation = 'pos')  # prints 0.928116


m, b = best_fit_slope_and_intercept(xs, ys)

# for x in xs:
#    regression_line.append((m * x) + b)
regression_line = [(m * x) + b for x in xs]

predict_x = 8 
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared) # prints 0.584415584416

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y) # prediction based on data
plt.plot(xs, regression_line)
plt.show
