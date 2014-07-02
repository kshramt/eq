import unittest

import numpy as np


HALF_PI = np.pi/2
N_REFLECT_MAX = 3




def _kernel_t_step(u, p):
        return u**2/_eta(u, p)


def _kernel_x_step(u, p):
    return p/_eta(u, p)


def _eta(u, p):
    return np.sqrt(u**2 - p**2)


def T_step(p, layers):
    hs, us = _parse_layers_T_X_step(layers)
    i_reflect = _i_reflect_T_X_step(p, us)
    if not(i_reflect is None):
        return 2*np.sum(h*_kernel_t_step(u, p) for h, u in zip(hs[0:i_reflect], us))


def X_step(p, layers):
    hs, us = _parse_layers_T_X_step(layers)
    i_reflect = _i_reflect_T_X_step(p, us)
    if not(i_reflect is None):
        return 2*np.sum(h*_kernel_x_step(u, p) for h, u in zip(hs[0:i_reflect], us))


def _i_reflect_T_X_step(p, us):
    for i_reflect, u in enumerate(us):
        if not _is_incident(p, u):
            return i_reflect


def _parse_layers_T_X_step(layers):
    hs = []
    us = []
    for h, v in layers:
        hs.append(h)
        us.append(1/v)
    return hs, us


def ray_path_1d_step(t, x, y, theta, is_down, layers):
    assert layers
    path = [(t, x, y)]
    assert 0 <= theta <= HALF_PI
    hs = [layer[0] for layer in layers]
    assert all(h >= 0 for h in hs)
    boundaries = _get_boundaries(hs)
    boundary_max = boundaries[-1]
    us = [1/layer[1] for layer in layers]
    assert all(u > 0 for u in us)
    assert 0 <= y <= boundary_max
    iy = _get_i_layer(boundaries, y)
    sin_theta = np.sin(theta)
    if is_down:
        boundary = boundaries[iy]
        if y < boundary: # trace a ray down to the lower boundary
            dy = boundary - y
            x += dy*np.tan(theta)
            u = us[iy]
            t += dy/np.cos(theta)*u
            y = boundary
            path.append((t, x, y))
            p = sin_theta*u
            if iy == len(hs):
                return {'p': p,
                        'path': path}
        else:
            if iy == len(hs):
                return {'p': sin_theta*us[iy],
                        'path': path}
            else:
                p = sin_theta*us[iy + 1]
        return {'p': p,
                'path': _ray_path_1d_step(path, p, is_down, iy, boundary_max, hs, us, 0)}
    else:
        boundary = boundaries[iy]
        u = us[iy]
        p = sin_theta*u
        if y < boundary: # trace a ray up to the upper boundary
            if iy == 0:
                t, x = _update_t_x_1d_step_up(t, x, y, theta, u)
                path.append((t, x, 0))
                return {'p': p,
                        'path': path}
            else:
                t, x = _update_t_x_1d_step_up(t, x, y - boundaries[iy - 1], theta, u)
                path.append((t, x, boundaries[iy - 1]))
                return {'p': p,
                        'path': _ray_path_1d_step(path, p, is_down, iy - 1, boundary_max, hs, us, 0)}
        else:
            return {'p': p,
                    'path': _ray_path_1d_step(path, p, is_down, iy, boundary_max, hs, us, 0)}


def _update_t_x_1d_step_up(t, x, abs_dy, theta, u):
    assert abs_dy >= 0
    return t + abs_dy/np.cos(theta)*u, x + abs_dy*np.tan(theta)


def _ray_path_1d_step(path, p, is_down, iy, boundary_max, hs, us, n_reflection):
    t, x, y = path[-1]

    if is_down:
        if y >= boundary_max:
            return path
        iys = range(iy + 1, len(hs))
        coef_dy = 1
        iy_reflection_modifier = -1
    else:
        if y <= 0:
            return path
        iys = range(iy, -1, -1)
        coef_dy = -1
        iy_reflection_modifier = +1

    for iy in iys:
        u = us[iy]
        if _is_incident(p, u):
            dy = coef_dy*hs[iy]
            t, x, y = _update_t_x_y_1d_step(t, x, y, dy, p, u)
            path.append((t, x, y))
        elif _is_reflection(p, u):
            n_reflection += 1
            if n_reflection >= N_REFLECT_MAX:
                return path
            else:
                return _ray_path_1d_step(path, p, not is_down, iy + iy_reflection_modifier, boundary_max, hs, us, n_reflection)
        else:
            return path
    return path


def _is_incident(p, u):
    return p < u


def _is_reflection(p, u):
    return p > u


def _update_t_x_y_1d_step(t, x, y, dy, p, u):
    abs_dy = np.absolute(dy)
    return t + abs_dy*_kernel_t_step(u, p), x + abs_dy*_kernel_x_step(u, p), y + dy


def _get_boundaries(hs):
    assert hs
    boundary = hs[0]
    boundaries = [boundary]
    for h in hs[1:]:
        boundary += h
        boundaries.append(boundary)
    return boundaries


def _get_i_layer(boundaries, y):
    assert boundaries
    assert 0 <= y <= boundaries[-1]
    return _get_i_layer_impl(boundaries, y)


def _get_i_layer_impl(boundaries, y):
    n = len(boundaries)
    if n >= 2:
        i = (n - 1)//2
        if boundaries[i] < y:
            return i + _get_i_layer_impl(boundaries[i + 1:], y) + 1
        else:
            return _get_i_layer_impl(boundaries[:i + 1], y)
    else:
        return 0


class Tester(unittest.TestCase):

    def test_T_X_step(self):
        hs = (1, 2, 3, 4, 5, 6)
        vs = (1, 1.1, 1.2, 1.3, 1.4, 100)
        layers = list(zip(hs, vs))
        p = 0.4
        theta = np.arcsin(p*vs[0])
        t, x, y = ray_path_1d_step(0, 0, 0, theta, True, layers)['path'][-1]
        self.assertAlmostEqual(T_step(p, layers), t)
        self.assertAlmostEqual(X_step(p, layers), x)
        self.assertAlmostEqual(0, y)

    def test_ray_path_1d_step(self):
        hs = (1, 2, 3, 4, 5, 6)
        # hs = (1, 2, 3, 4, 5, 6)
        vs = (1, 1.1, 1.2, 1.3, 1.4, 100)
        layers = list(zip(hs, vs))
        theta = HALF_PI/2
        p = np.sin(theta)/vs[1]

        # down
        path = ray_path_1d_step(1, 3, 2, theta, True, layers)['path']
        t, x, y = path[4]
        self.assertAlmostEqual(t,
                               1 +
                               1/vs[1]**2/_eta(1/vs[1], p) +
                               3/vs[2]**2/_eta(1/vs[2], p) +
                               4/vs[3]**2/_eta(1/vs[3], p) +
                               5/vs[4]**2/_eta(1/vs[4], p))
        self.assertAlmostEqual(x,
                               3 +
                               p*
                               (1/_eta(1/vs[1], p) +
                                3/_eta(1/vs[2], p) +
                                4/_eta(1/vs[3], p) +
                                5/_eta(1/vs[4], p)))
        self.assertAlmostEqual(y, 15)

        # up
        path = ray_path_1d_step(1, 3, 19, theta, False, list(reversed(layers)))['path']
        t, x, y = path[4]
        self.assertAlmostEqual(t,
                               1 +
                               1/vs[1]**2/_eta(1/vs[1], p) +
                               3/vs[2]**2/_eta(1/vs[2], p) +
                               4/vs[3]**2/_eta(1/vs[3], p) +
                               5/vs[4]**2/_eta(1/vs[4], p))
        self.assertAlmostEqual(x,
                               3 +
                               p*
                               (1/_eta(1/vs[1], p) +
                                3/_eta(1/vs[2], p) +
                                4/_eta(1/vs[3], p) +
                                5/_eta(1/vs[4], p)))
        self.assertAlmostEqual(y, 6)


if __name__ == '__main__':
    unittest.main()
