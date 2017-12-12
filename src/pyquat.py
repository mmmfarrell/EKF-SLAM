import numpy as np

cross_matrix = np.array([[[0, 0, 0],
                          [0, 0, -1.0],
                          [0, 1.0, 0]],
                         [[0, 0, 1.0],
                          [0, 0, 0],
                          [-1.0, 0, 0]],
                         [[0, -1.0, 0],
                          [1, 0, 0],
                          [0, 0, 0]]])

qmat_matrix = np.array([[[1.0, 0, 0, 0],
                         [0, -1.0, 0, 0],
                         [0, 0, -1.0, 0],
                         [0, 0, 0, -1.0]],
                        [[0, 1.0, 0, 0],
                         [1.0, 0, 0, 0],
                         [0, 0, 0, 1.0],
                         [0, 0, -1.0, 0]],
                        [[0, 0, 1.0, 0],
                         [0, 0, 0, -1.0],
                         [1.0, 0, 0, 0],
                         [0, 1.0, 0, 0]],
                        [[0, 0, 0, 1.0],
                         [0, 0, 1.0, 0],
                         [0, -1.0, 0, 0],
                         [1.0, 0, 0, 0]]])

def quat_arr_to_euler(array):
    assert array.shape[0] == 4
    w = array[0,:]
    x = array[1,:]
    y = array[2,:]
    z = array[3,:]
    return np.array([np.arctan2(2.0 * (w * x + y * z), 1. - 2. * (x * x + y * y)),
                     np.arcsin(2.0 * (w * y - z * x)),
                     np.arctan2(2.0 * (w * z + x * y), 1. - 2. * (y * y + z * z))])



def skew(v):
    assert v.shape == (3, 1)
    return cross_matrix.dot(v).squeeze()

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))

class Quaternion():
    def __init__(self, v):
        assert isinstance(v, np.ndarray)
        assert v.shape == (4,1)
        self.arr = v

    def __str__(self):
        return "[ " + str(self.arr[0,0]) + ", " + str(self.arr[1,0]) + "i, " \
               + str(self.arr[2,0]) + "j, " + str(self.arr[3,0]) + "k ]"

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return self.otimes(other)

    def __imul__(self, other):
        self.arr = qmat_matrix.dot(other.arr).squeeze().dot(self.arr)
        return self

    def __add__(self, other):
        return self.boxplus(other)

    def __iadd__(self, other):
        assert other.shape == (3, 1)
        delta = other.copy()

        norm_delta = norm(delta)

        # If we aren't going to run into numerical issues
        if norm_delta > 1e-4:
            v = np.sin(norm_delta / 2.) * (delta / norm_delta)
            self.arr = qmat_matrix.dot(np.vstack((np.cos(norm_delta/2.0), v))).squeeze().dot(self.arr)
        else:
            arrdot = 0.5*qmat_matrix.dot(np.vstack((np.zeros((1,1)), delta))).squeeze().dot(self.arr)
            self.arr += arrdot
            self.arr /= norm(self.arr)
        return self

    def __sub__(self, other):
        dq = other.inverse.otimes(self)
        if dq.w < 0.0:
            dq.arr *= -1.0
        return self.log(dq)

    @staticmethod
    def __test__():
        # Make sure that active rotations are active rotations
        v = np.array([[0, 0, 1.]]).T
        v_active_rotated = np.array([[0, -0.5**0.5, 0.5**0.5]]).T
        beta = np.array([[1., 0, 0]]).T
        q_x_45 = Quaternion.from_axis_angle(beta, 45.0 * np.pi / 180.0)
        assert norm(q_x_45.rot(v) - v_active_rotated) < 1e-8

        # And that passive rotations are passive rotations
        v_passive_rotated = np.array([[0, 0.5 ** 0.5, 0.5 ** 0.5]]).T
        assert norm(q_x_45.invrot(v) - v_passive_rotated) < 1e-8

        import pyquaternion
        for i in range(100):
            v = np.random.uniform(-100, 100, (3,1))
            v_small = np.random.normal(-1/4., 1/4., (3,1))
            q = Quaternion(np.random.uniform(-1, 1, (4,1)))
            q.normalize()
            q2 = Quaternion(np.random.uniform(-1, 1, (4,1)))
            q2.normalize()

            # Check against oracle
            # (pyquaternion returns the active rotation matrix because it is stupid)
            oracle_q = pyquaternion.Quaternion(q.arr)
            oracle_q2 = pyquaternion.Quaternion(q2.arr)
            assert norm(oracle_q.rotation_matrix.T - q.R) < 1e-8 # make sure they create the same rotation matrix
            assert norm((oracle_q2 * oracle_q).elements[:,None] - (q2 * q).elements) < 1e-8 # make sure they do the same thing for quat multiplication

            # Check equivalence of rot, invrot and R
            assert norm(q.rot(v) - q.R.T.dot(v)) < 1e-8
            assert norm(q.invrot(v) - q.R.dot(v)) < 1e-8

            # Check that rotations are inverses of each other
            assert norm(q.rot(q.invrot(v)) - v) < 1e-8
            assert norm(q.invrot(q.rot(v)) - v) < 1e-8
            assert norm(q.R.dot(q.R.T.dot(v)) - v) < 1e-8
            assert norm(q.R.T.dot(q.R.dot(v)) - v) < 1e-8

            # Check from_two_vectors
            v1 = np.random.uniform(-100, 100, (3,1))
            v2 = np.random.uniform(-100, 100, (3,1))
            v1 /= norm(v1)
            v2 /= norm(v2)
            assert norm(Quaternion.from_two_unit_vectors(v1, v2).rot(v1) - v2) < 1e-8
            assert norm(Quaternion.from_two_unit_vectors(v2, v1).invrot(v1)  - v2) < 1e-8

            # Check from_R
            R = q.R
            qR = Quaternion.from_R(R)
            assert norm(qR.rot(v) - R.T.dot(v)) < 1e-8

            assert norm((q*q.inverse).elements - np.array([[1., 0, 0, 0]]).T) < 1e-8

            # Check that qexp is right by comparing with rotation matrix qexp and axis-angle
            import scipy.linalg
            omega = np.random.uniform(-1, 1, (3,1))
            R_omega_exp = scipy.linalg.expm(skew(omega))
            q_R_omega_exp = Quaternion.from_R(R_omega_exp.T)
            q_omega = Quaternion.from_axis_angle(omega/norm(omega), norm(omega))
            q_omega_exp = Quaternion.exp(omega)
            assert norm(q_R_omega_exp.elements - q_omega.elements) < 1e-8
            assert norm(q_omega_exp.elements - q_omega.elements) < 1e-8

            # Check qexp and qlog are the inverses of each other
            assert norm(Quaternion.log(Quaternion.exp(v_small)) - v_small) < 1e-8
            assert norm(Quaternion.exp(v_small).elements) - 1.0 < 1e-8

            # Check boxplus and boxminus
            delta1 = np.random.normal(-0.25, 0.25, (3,1))
            delta2 = np.random.normal(-0.25, 0.25, (3, 1))
            assert norm((q + np.zeros((3,1))).elements - q.elements) < 1e-8
            assert norm((q + (q2 - q)).elements - q2.elements) < 1e-8 or norm((q + (q2 - q)).elements + q2.elements) < 1e-8
            assert norm(((q + delta1) - q) - delta1) < 1e-8
            assert norm((q + delta1) - (q + delta1)) <= norm(delta1 - delta2)

            # Check iadd and imul
            qcopy = q.copy()
            qcopy += delta1
            assert norm(qcopy.elements - (q+delta1).elements) < 1e-8
            qcopy = q.copy()
            qcopy *= q2
            assert norm(qcopy.elements - (q * q2).elements) < 1e-8

        print "pyquat test [PASSED]"

    @property
    def w(self):
        return self.arr[0,0]

    @property
    def x(self):
        return self.arr[1, 0]

    @property
    def y(self):
        return self.arr[2, 0]

    @property
    def z(self):
        return self.arr[3, 0]

    # returns in [w, x, y, z]
    @property
    def elements(self):
        return self.arr

    @property
    def euler(self):
        w = self.arr[0, 0]
        x = self.arr[1, 0]
        y = self.arr[2, 0]
        z = self.arr[3, 0]
        return np.array([[np.arctan2(2.0*(w*x + y*z), 1. - 2.*(x*x + y*y))],
                         [np.arcsin(2.0*(w*y - z*x))],
                         [np.arctan2(2.0*(w*z + x*y), 1. - 2.*(y*y + z*z))]])


    # Calculates the rotation matrix equivalent.  If you are performing a rotation,
    # is is much faster to use rot or invrot
    @property
    def R(self):
        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]

        wx = w*x
        wy = w*y
        wz = w*z
        xx = x*x
        xy = x*y
        xz = x*z
        yy = y*y
        yz = y*z
        zz = z*z

        return np.array([[1. - 2.*yy - 2.*zz, 2.*xy + 2.*wz, 2.*xz - 2.*wy],
                         [2.*xy - 2.*wz, 1. - 2.*xx - 2.*zz, 2.*yz + 2.*wx],
                         [2.*xz + 2.*wy, 2.*yz - 2.*wx, 1. - 2.*xx - 2.*yy]])

    # Calculates the quaternion exponential map for a 3-vector.
    # Returns a quaternion
    @staticmethod
    def exp(v):
        assert v.shape == (3,1)
        delta = v.copy()

        norm_delta = norm(delta)

        if norm_delta > 1e-4:
            q_exp = np.vstack((np.array([[np.cos(norm_delta/2.0)]]), np.sin(norm_delta/2.0)*delta/norm_delta))
        else:
            q_exp = np.vstack((np.array([[1.0]]), delta/2.0))
            q_exp/=norm(q_exp)
        return Quaternion(q_exp)

    @staticmethod
    def Identity():
        return Quaternion(np.array([[1.0, 0, 0, 0]]).T)

    @staticmethod
    def random():
        arr = np.random.uniform(-1, 1, (4,1))
        arr /= norm(arr)
        return Quaternion(arr)

    @staticmethod
    def log(q):
        assert isinstance(q, Quaternion)

        v = q.arr[1:]
        w = q.arr[0,0]
        norm_v = norm(v)
        if norm_v < 1e-8:
            return np.zeros((3,1))
        else:
            return 2.0*np.arctan2(norm_v, w)*v/norm_v

    def copy(self):
        q_copy = Quaternion(self.arr.copy())
        return q_copy

    def normalize(self):
        self.arr /= norm(self.arr)

    # Perform an active rotation on v (same as q.R.T.dot(v), but faster) CONFIRMED
    def rot(self, v):
        assert v.shape[0] == 3
        skew_xyz = skew(self.arr[1:])
        t = 2.0 * skew_xyz.dot(v)
        out = v + self.arr[0,0] * t + skew_xyz.dot(t)
        return out

    # Perform a passive rotation on v (same as q.R.dot(v), but faster) CONFIRMED
    def invrot(self, v):
        assert v.shape[0] == 3
        skew_xyz = skew(self.arr[1:])
        t = 2.0 * skew_xyz.dot(v)
        return v - self.arr[0,0] * t + skew_xyz.dot(t)

    def inv(self):
        self.arr[1:] *= -1.0

    @property
    def inverse(self):
        inverted = self.arr.copy()
        inverted[1:] *= -1.0
        return Quaternion(inverted)

    # Calculates the quaternion which rotates u into v.
    # That is, if q = q_from_two_unit_vectors(u,v)
    # q.rot(u) = v and q.invrot(v) = u
    @staticmethod
    def from_two_unit_vectors(u, v):
        assert u.shape == (3,1)
        assert v.shape == (3,1)
        u = u.copy()
        v = v.copy()

        arr = np.array([[1., 0., 0., 0.]]).T

        d = u.T.dot(v).squeeze()
        if d < 1.0:
            invs = (2.0*(1.0+d))**-0.5
            xyz = skew(u).dot(v)*invs.squeeze()
            arr[0,0]=0.5/invs
            arr[1:,:] = xyz
            arr /= norm(arr)
        return Quaternion(arr)

    @staticmethod
    def from_R(m):
        q = np.zeros((4,1))
        tr = np.trace(m)

        if tr > 0:
            S = np.sqrt(tr+1.0) * 2.
            q[0] = 0.25 * S
            q[1] = (m[1,2] - m[2,1]) / S
            q[2] = (m[2,0] - m[0,2]) / S
            q[3] = (m[0,1] - m[1,0]) / S
        elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.
            q[0] = (m[1,2] - m[2,1]) / S
            q[1] = 0.25 * S
            q[2] = (m[1,0] + m[0,1]) / S
            q[3] = (m[2,0] + m[0,2]) / S
        elif m[1,1] > m[2,2]:
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.
            q[0] = (m[2,0] - m[0,2]) / S
            q[1] = (m[1,0] + m[0,1]) / S
            q[2] = 0.25 * S
            q[3] = (m[2,1] + m[1,2]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.
            q[0] = (m[0,1] - m[1,0]) / S
            q[1] = (m[2,0] + m[0,2]) / S
            q[2] = (m[2,1] + m[1,2]) / S
            q[3] = 0.25 * S
        return Quaternion(q)

    @staticmethod
    def from_axis_angle(axis, angle):
        assert axis.shape == (3,1) and isinstance(angle, float)
        alpha_2 = np.array([[angle/2.0]])
        return Quaternion(np.vstack((np.cos(alpha_2), axis*np.sin(alpha_2))))


    @staticmethod
    def from_euler(roll, pitch, yaw):
        cp = np.cos(roll/2.0)
        ct = np.cos(pitch/2.0)
        cs = np.cos(yaw/2.0)
        sp = np.sin(roll/2.0)
        st = np.sin(pitch/2.0)
        ss = np.sin(yaw/2.0)

        return Quaternion(np.array([[cp*ct*cs - sp*st*ss],
                                    [sp*st*cs + cp*ct*ss],
                                    [sp*ct*cs + cp*st*ss],
                                    [cp*st*cs - sp*ct*ss]]))

    def otimes(self, q):
        q_new = Quaternion(qmat_matrix.dot(q.arr).squeeze().dot(self.arr).copy())
        return q_new

    def boxplus(self, delta):
        assert delta.shape == (3,1)
        return self.otimes(Quaternion.exp(delta))

if __name__ == '__main__':
    Quaternion.__test__()
