import numpy as np

def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x +b)) ** 2
    return total_error / len(points)


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iterations):
        # update b and m with the new more accurate b and m by performing
        # this gradient step
        b, m = step_gradient(b, m, points, learning_rate)

    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):

    # starting points for our gradient
    b_gradient = 0
    m_gradient = 0

    n = len(points)

    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]

        # direction with respect to b and m
        # computing partial derivatives of our error fn
        b_gradient += (2/n) * -(y - ((m_current * x) + b_current))
        m_gradient += (2/n) * -x * (y - ((m_current * x) + b_current))

    # update our b and m values using partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]




def run():

    # Step 1: Collect Data
    points = np.genfromtxt('data.csv', delimiter=',')

    # Step 2: Define our HyperParameters

    # how fast should our model converge
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Step 3: train our model
    initial_error = compute_error_for_line_given_points(initial_b, initial_m, points)
    print('starting gradient descent at b={0}, m={1}, error={2}'.format(initial_b, initial_m, initial_error))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    ending_error = compute_error_for_line_given_points(b, m, points)
    print('ending point at b={1}, m={2}, error={3}'.format(num_iterations, b, m, ending_error))




if __name__ == '__main__':
    run()
