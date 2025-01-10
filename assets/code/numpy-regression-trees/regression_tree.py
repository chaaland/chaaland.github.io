from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class RegressionTree(BaseEstimator):
    def __init__(
        self,
        min_sample: int = 5,
        max_depth: int = 10,
    ):
        """Initialise the regression tree

        :param min_sample: an internal node can be split only if it contains more than min_sample points
        :param max_depth: restriction of tree depth
        """
        self.min_sample = min_sample
        self.max_depth = max_depth
        self._is_fit = False
        self._value = None

    # @profile
    def _split_quality(self, y_left: np.array, y_right: np.array):
        """Return the quality of the proposed node split

        :param y_left: target values in the proposed left node
        :param y_right: target values in the proposed right node
        :return: the weighted sum of variances of the left and right nodes
        """
        n_left, n_right = y_left.size, y_right.size
        n_tot = n_left + n_right
        w_left, w_right = n_left / n_tot, n_right / n_tot

        return w_left * np.var(y_left) + w_right * np.var(y_right)

    # @profile
    def _leaf_value_estimator(self, y: np.array):
        """Return a prediction for the values contained in the leaf

        :param y: array of values in the leaf
        :return: scalar prediction value
        """
        return np.mean(y)

    # @profile
    def _optimal_split_fast(self, feat_values: np.array, y: np.array):
        n_data = y.size
        best_split_score, best_split_threshold = np.inf, np.inf

        sort_indices = np.argsort(feat_values)
        sorted_values = feat_values[sort_indices]
        sorted_labels = y[sort_indices].squeeze()
        _, unique_indexes = np.unique(sorted_values, return_index=True)
        sum_left, sum_right = 0, np.sum(y)
        sum_square_left, sum_square_right = 0, np.sum(np.square(y))

        for i, curr_split in enumerate(unique_indexes):
            prev_split = unique_indexes[i - 1] if i - 1 > 0 else 0
            sum_delta = np.sum(sorted_labels[prev_split:curr_split])
            sum_square_delta = np.sum(sorted_labels[prev_split:curr_split] ** 2)

            sum_left += sum_delta
            sum_right -= sum_delta
            sum_square_left += sum_square_delta
            sum_square_right -= sum_square_delta
            if curr_split < self.min_sample or curr_split > n_data - self.min_sample:
                continue

            n_left = curr_split
            n_right = n_data - n_left
            w_left, w_right = n_left / n_data, n_right / n_data
            var_left = (sum_square_left / n_left) - (sum_left / n_left) ** 2
            var_right = (sum_square_right / n_right) - (sum_right / n_right) ** 2
            split_score = w_left * var_left + w_right * var_right

            if split_score < best_split_score:
                best_split_score = split_score
                best_split_threshold = (sorted_values[prev_split] + sorted_values[curr_split]) / 2

        return best_split_score, best_split_threshold

    # @profile
    def _optimal_split(self, feat_values: np.array, y: np.array):
        n_data = y.size
        best_split_score, best_split_threshold = np.inf, np.inf

        sort_indices = np.argsort(feat_values)
        sorted_values = feat_values[sort_indices]
        sorted_labels = y[sort_indices]
        _, unique_indexes = np.unique(sorted_values, return_index=True)
        for i in unique_indexes:
            if i < self.min_sample or i > n_data - self.min_sample:
                continue

            y_l, y_r = sorted_labels[:i], sorted_labels[i:]
            split_score = self._split_quality(y_l, y_r)
            if split_score < best_split_score:
                best_split_score = split_score
                best_split_threshold = (sorted_values[i] + sorted_values[i - 1]) / 2

        return best_split_score, best_split_threshold

    # @profile
    def fit(self, X: np.array, y: np.array):
        """Fit the decision tree in place

        This should fit the tree by setting the values:
        self.split_id (the index of the feature we want to split on, if we're splitting)
        self.split_value (the corresponding value of that feature where the split is)
        self._value (the prediction value if the tree is a leaf node).

        If we are splitting the node, we should also init self.left and self.right to
        be RegressionTree objects corresponding to the left and right subtrees. These
        subtrees should be fit on the data that fall to the left and right, respectively,
        of self.split_value. This is a recursive tree building procedure.

        :param X: a numpy array of training data, shape = (n, p)
        :param y: a numpy array of labels, shape = (n,)
        :return: self
        """
        n_data, n_features = X.shape
        self._is_fit = True
        if self.max_depth == 0 or n_data <= self.min_sample:
            self._value = self._leaf_value_estimator(y)
            return self

        best_split_feature, best_split_score = 0, np.inf
        for feature in range(n_features):
            feat_values = X[:, feature]
            score, threshold = self._optimal_split_fast(feat_values, y)
            # score, threshold = self._optimal_split(feat_values, y)

            if score < best_split_score:
                best_split_score = score
                best_split_threshold = threshold
                best_split_feature = feature

        if np.isinf(best_split_score):
            self._value = self._leaf_value_estimator(y)
            return self

        mask = X[:, best_split_feature] < best_split_threshold

        self.split_id = best_split_feature
        self.split_value = best_split_threshold

        X_l, y_l = X[mask, :], y[mask]
        X_r, y_r = X[~mask, :], y[~mask]

        self.left = RegressionTree(
            min_sample=self.min_sample,
            max_depth=self.max_depth - 1,
        )
        self.right = RegressionTree(
            min_sample=self.min_sample,
            max_depth=self.max_depth - 1,
        )

        self.left.fit(X_l, y_l)
        self.right.fit(X_r, y_r)

        return self

    def _predict_instance(self, x: np.ndarray):
        """Predict value by regression tree

        :param x: numpy array with new data, shape (p,)
        :return: prediction
        """
        if not self._is_fit:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        elif self._value is not None:
            return self._value
        elif x[self.split_id] <= self.split_value:
            return self.left._predict_instance(x)
        else:
            return self.right._predict_instance(x)

    def predict(self, X: np.ndarray):
        n_predict, _ = X.shape
        y_pred = np.empty(n_predict)

        for i, x in enumerate(X):
            y_pred[i] = self._predict_instance(x)

        return y_pred


# @profile
def profiling_example():
    X_train = np.linspace(0, np.pi, 10000).reshape(-1, 1)
    y_train = np.cos(X_train**2)

    reg_tree = RegressionTree(min_sample=5, max_depth=10)
    reg_tree.fit(X_train, y_train)


def plot_2d_example():
    X_train = np.linspace(0, np.pi, 500).reshape(-1, 1)
    y_train = np.cos(X_train**2)

    fig, ax = plt.subplots()
    ax.plot(X_train, y_train)
    ax.set_ylim(-1.05, 1.05)
    ax.set(xlabel="x", ylabel="y", title=r"$y = \cos(x^2)$")
    plt.savefig("../../images/numpy-regression-trees/1d-chirp.png")

    def plot_tree_model_fit(max_depth):
        reg_tree = RegressionTree(min_sample=5, max_depth=max_depth)
        reg_tree.fit(X_train, y_train)

        x_vals = np.linspace(0, np.pi, 1000).reshape(-1, 1)
        y_vals = reg_tree.predict(x_vals)

        fig, ax = plt.subplots()
        ax.step(x_vals, y_vals)
        ax.set(xlabel="x", ylabel="y", title=f"Regression Tree: max-depth={max_depth}, min-samples=5")

        ax.set_ylim(-1.05, 1.05)

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    imageio.mimsave(
        "../../gifs/numpy-regression-trees/1d-regression-tree.gif",
        [plot_tree_model_fit(i) for i in range(2, 16)],
        fps=3,
    )


def plot_3d_example():
    x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    R_mesh = np.sqrt(X_mesh**2 + Y_mesh**2 + 1e-5)
    Z_mesh = np.sin(R_mesh) / R_mesh

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap="magma", edgecolor="none")
    ax.set(xlabel="x", ylabel="y", title=r"$y = \sin(\sqrt{x^2+y^2}/\sqrt{x^2+y^2})$")
    plt.savefig("../../images/numpy-regression-trees/2d-sinc.png")

    X_train = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
    y_train = Z_mesh.ravel()

    def plot_tree_model_fit(max_depth):
        reg_tree = RegressionTree(min_sample=5, max_depth=max_depth)
        reg_tree.fit(X_train, y_train)

        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X_pred_mesh, Y_pred_mesh = np.meshgrid(x_vals, y_vals)

        X_pred = np.column_stack([X_pred_mesh.ravel(), Y_pred_mesh.ravel()])
        z_pred = reg_tree.predict(X_pred)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(X_pred_mesh, Y_pred_mesh, z_pred.reshape(X_pred_mesh.shape), cmap="magma", edgecolor="none")
        ax.set(xlabel="x", ylabel="y", title=f"Regression Tree: max-depth={max_depth}, min-samples=5")

        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-2 * np.pi, 2 * np.pi)
        ax.set_zlim(-0.2, 1.0)

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    imageio.mimsave(
        "../../gifs/numpy-regression-trees/2d-regression-tree.gif",
        [plot_tree_model_fit(i) for i in range(2, 16)],
        fps=3,
    )


def plot_splash_image():
    np.random.seed(3)
    n = 100
    eps = 0.2 * np.random.randn(n)
    x_train = np.random.randn(n)
    y_train = np.square(x_train) + eps
    small_tree = RegressionTree(min_sample=5, max_depth=3)
    med_tree = RegressionTree(min_sample=5, max_depth=5)
    big_tree = RegressionTree(min_sample=5, max_depth=11)

    small_tree.fit(x_train[:, np.newaxis], y_train)
    med_tree.fit(x_train[:, np.newaxis], y_train)
    big_tree.fit(x_train[:, np.newaxis], y_train)

    x_vals = np.linspace(-3, 3, 1000)[:, np.newaxis]
    y_small_pred = small_tree.predict(x_vals)
    y_med_pred = med_tree.predict(x_vals)
    y_big_pred = big_tree.predict(x_vals)

    plt.scatter(x_train, y_train, color="k", alpha=0.7)
    plt.step(x_vals, y_small_pred)
    plt.step(x_vals, y_med_pred)
    plt.step(x_vals, y_big_pred)

    fname = Path("..", "..", "images", "numpy-regression-trees", "splash-image.png")
    plt.savefig(fname)


if __name__ == "__main__":
    # matplotlib.use("TkAgg")
    import imageio
    import matplotlib.pyplot as plt

    # profiling_example()
    # plot_2d_example()
    # plot_3d_example()
    plot_splash_image()
