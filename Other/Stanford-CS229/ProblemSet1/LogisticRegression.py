import numpy as np
import src.util as util

from src.linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        def h(theta,x):
            """ Implementation of the hypothesis for logistic regression.
            Args:
                theta: Represents our paramentes having shape (n,)
                 x: Training example inputs. Shape (m, n).
            """
            return 1 / (1+np.exp(-np.dot(x,theta)))

        def gradient(theta,x, y):
            
            """Vectorized implementation of the gradient of J(theta).

            :param theta: Shape (n,).
            :param x:     All training examples of shape (m, n).
            :param y:     All labels of shape (m,).
            :return:      The gradient of shape (n,).
            """
            m,_ = x.shape
            return -1/m * np.dot(x.T, y - h(theta,x) )
        
        
        def hessian(theta, x):
            
            """Vectorized implementation of the Hessian of J(theta).

            :param theta: Shape (n,).
            :param x:     All training examples of shape (m, n).
            :return:      The Hessian of shape (n, n).
            """
            m, _ = x.shape
            h_theta_x = np.reshape(h(theta,x), (-1,1))
            return 1 / m * np.dot(x.T, h_theta_x * (1-h_theta_x)* x)
        
        def next_theta(theta,x,y):
            """The next theta updated by Newton's Method.

            :param theta: Shape (n,).
            :return:      The updated theta of shape (n,).
            """
            return theta - np.dot(np.linalg.inv(hessian(theta,x)), gradient(theta,x,y))

        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
            

        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta,x,y)
        self.theta = new_theta


    def fit_gd(self, x, y):
        
        """Run Gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        
        def h(theta,x):
            """ Implementation of the hypothesis for logistic regression.
            Args:
                theta: Represents our paramentes having shape (n,)
                 x: Training example inputs. Shape (m, n).
            """
            return 1 / (1+np.exp(-np.dot(x,theta)))
        def gradient(theta, x, y):
            """Vectorized implementation of the gradient of J(theta).

            :param theta: Shape (n,).
            :param x:     All training examples of shape (m, n).
            :param y:     All labels of shape (m,).
            :return:      The gradient of shape (n,).
            """
            
            m,_ = x.shape
            return 1/m * np.dot(x.T, y-h(theta, x))

        def update_theta(theta,x,y, learning_rate = 0.001):
            """Update the value of theta according to the gradient descent algorithm
            :param theta: Shape (n,).
            :return:      The updated theta of shape (n,).
            """
            
            return theta + gradient(theta, x, y)*learning_rate

        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
  

        old_theta = self.theta
        new_theta = update_theta(self.theta, x, y)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = update_theta(old_theta,x,y)

        self.theta = new_theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return x @ self.theta >= 0
        # *** END CODE HERE ***
