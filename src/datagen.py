import numpy


def generate_ou_2d_process(n_realizations, n_steps, dt, B, Sigma, x0=None):
    """
    Generate realizations of a 2D Ornstein-Uhlenbeck process using the Euler-Maruyama method.

    Parameters:
    - n_realizations (int): Number of realizations to generate.
    - n_steps (int): Number of time steps for each realization.
    - dt (float): Time step size.
    - B (numpy.ndarray): 2x2 drift matrix.
    - Sigma (numpy.ndarray): 2x2 diffusion matrix.
    - x0 (numpy.ndarray, optional): Initial state. If None, starts at [0, 0].

    Returns:
    - List[numpy.ndarray]: List of arrays, each of shape (n_steps, 2) containing a simulated process.
    """
    # The code might work even if B and Sigma are not 2x2, but we assert here to make sure
    assert B.shape == (2, 2)
    assert Sigma.shape == (2, 2)
    
    if x0 is None:
        x0 = numpy.zeros(2)
    
    # Precompute the square root of Sigma * dt
    sqrt_Sigma_dt = numpy.linalg.cholesky(Sigma * dt)
    
    # Initialize the output list
    X = list()
    
    for _ in range(n_realizations):
        X_i = numpy.zeros((n_steps, 2))
        X_i[0] = x0
        
        for t in range(1, n_steps):
            dW = numpy.random.normal(0, 1, 2)
            X_i[t] = X_i[t-1] + numpy.dot(B, -X_i[t-1]) * dt + numpy.dot(sqrt_Sigma_dt, dW)
        
        X.append(X_i)
    
    return X


def _test_generation():
    import matplotlib.pyplot as plt

    fs = 1000
    B = 2 * numpy.array([[1.0, 0.0], [0.0, 1.0]])
    Sigma = numpy.array([[1, 0.5], [0.5, 1]])
    x_t = generate_ou_2d_process(n_realizations=1, n_steps=500, dt=1/fs, B=B, Sigma=Sigma)

    fs = 1000
    B = numpy.array([[9.46, 3.27], [3.27, 6.5]])
    Sigma = numpy.array([[1, 0.5], [0.5, 1]])
    z_t = generate_ou_2d_process(n_realizations=1, n_steps=500, dt=1/fs, B=B, Sigma=Sigma)

    plt.plot(x_t[0][:, 0], x_t[0][:, 1], label="2D OU Process #1")
    plt.plot(z_t[0][:, 0], z_t[0][:, 1], label="2D OU Process #2")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Ornstein-Uhlenbeck Process')
    
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    _test_generation()