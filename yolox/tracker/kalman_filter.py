# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    Two motion models, selected via `motion_model`:

      'cv' (default) -- constant velocity. The 8-dimensional state space
          x, y, a, h, vx, vy, va, vh
      'ca' -- constant acceleration (2026-07-29, Prof Farzad follow-up on the
          two-frame IoU gate: does a t-2/t-1/t coast prediction help once the
          filter's own motion model has a use for the third point?). The
          12-dimensional state space
          x, y, a, h, vx, vy, va, vh, ax, ay, aa, ah

    Either way (x, y) is the bounding box center position, a the aspect ratio,
    h the height, and the rest their derivatives. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, motion_model='cv'):
        ndim, dt = 4, 1.
        if motion_model not in ('cv', 'ca'):
            raise ValueError("motion_model must be 'cv' or 'ca'")
        self.motion_model = motion_model
        self.order = 2 if motion_model == 'ca' else 1  # # of derivative blocks beyond position
        state_dim = ndim * (1 + self.order)

        # Create Kalman filter model matrices.
        #   pos  += vel*dt [+ 0.5*acc*dt^2 if CA]
        #   vel  += acc*dt                [CA only]
        #   acc  unchanged                [CA only]
        self._motion_mat = np.eye(state_dim, state_dim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        if self.order == 2:
            for i in range(ndim):
                self._motion_mat[i, 2 * ndim + i] = 0.5 * dt * dt
                self._motion_mat[ndim + i, 2 * ndim + i] = dt
        self._update_mat = np.eye(ndim, state_dim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        # CA only. Originally set by analogy to the velocity weight (never swept) -- overridable
        # via DARE_KF_ACCEL_NOISE for the 2026-07-31 hyperparameter sweep.
        self._std_weight_acceleration = float(os.environ.get('DARE_KF_ACCEL_NOISE', 1. / 160))

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8- or 12-dimensional, depending on
            motion_model) and covariance matrix of the new track. Unobserved
            velocities/accelerations are initialized to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        parts = [mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        if self.order == 2:
            mean_acc = np.zeros_like(mean_pos)
            parts.append(mean_acc)
            std += [
                10 * self._std_weight_acceleration * measurement[3],
                10 * self._std_weight_acceleration * measurement[3],
                1e-5,
                10 * self._std_weight_acceleration * measurement[3]]
        mean = np.r_[tuple(parts)]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The mean vector of the object state at the previous time step
            (8-dim for 'cv', 12-dim for 'ca').
        covariance : ndarray
            The covariance matrix of the object state at the previous time
            step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        std_list = std_pos + std_vel
        if self.order == 2:
            std_acc = [
                self._std_weight_acceleration * mean[3],
                self._std_weight_acceleration * mean[3],
                1e-5,
                self._std_weight_acceleration * mean[3]]
            std_list = std_list + std_acc
        motion_cov = np.diag(np.square(np.array(std_list)))

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        parts = std_pos + std_vel
        if self.order == 2:
            std_acc = [
                self._std_weight_acceleration * mean[:, 3],
                self._std_weight_acceleration * mean[:, 3],
                1e-5 * np.ones_like(mean[:, 3]),
                self._std_weight_acceleration * mean[:, 3]]
            parts = parts + std_acc
        sqr = np.square(np.vstack(parts)).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')