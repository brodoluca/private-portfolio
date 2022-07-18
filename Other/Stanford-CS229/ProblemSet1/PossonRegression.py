import numpy as np
import src.util

from src.linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def h(self,theta, x):
            return np.exp(x @ theta) # try to invert these two if the algorithm works

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        def h(theta, x):
            return np.exp(x @ theta) # try to invert these two if the algorithm works
        
        
        def next_step(theta, x, y):
            return self.step_size* np.dot(x.T,y-h(theta,x)) / m
        m, n = x.shape
        # Initialize theta
        if self.theta is None:
            theta = np.zeros(n)
        else:
            theta = self.theta
        
        step = next_step(theta,x,y)
        while np.linalg.norm(step, 1) >= self.eps:
            theta += step 
            step = next_step(theta,x,y)
            
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        
        return self.h(self.theta, x)
        
        # *** END CODE HERE ***