import numpy as np
import util

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.

    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}_train.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval, y_eval)
    util.plot(x_eval, y_eval, model.theta, 'output/p01b_{}_eval.png'.format(pred_path[-5]))

    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)

    guess random theta
    while abs(theta_new-theta_old) < error:
        calculate gradient
        calculate hessian
        update theta w/ gradient and hessian
    """
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        m, n = x.shape
        self.theta = np.zeros(n) # (3,)

        while True:
            theta_old = self.theta.copy()

            s = self.sigmoid(x @ self.theta) # (800, 1)

            l_grad = x.T @ (s - y) / m # (3, 800) @ (800,) -> (3,)

            s_deriv = s * (1 - s)
            hess = (x.T * s_deriv).dot(x) / m # (3, 1)
            self.theta -= np.linalg.inv(hess) @ l_grad
            if np.linalg.norm(self.theta - theta_old) < self.eps:
                l = np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))
                print("Final Training Log Likelihood: ", l)
                break

        # *** END CODE HERE ***

    def predict(self, x, y):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        s = self.sigmoid(x @ self.theta)
        l = np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))

        print("Final Eval Log Likelihood: ", l)

        return s
        # *** END CODE HERE ***