# `eq`

## Build

```bash
make build
```

## Test

```bash
make check
```

## `eq.moment_tensor`

### Output focal mechanisms in VTK format for 3D visualization

```bash
python3 <<EOF >| meca.vtk
import math
import eq.moment_tensor

m = eq.moment_tensor.MomentTensor()

x1, y1, z1 = 1, 2, 3
m.strike_dip_rake = 42, 80, 15
point1s, triangle1s, amplitude1s = m.amplitude_distribution(4)
points1 = [point1 + [x1, y1, z1] for point1 in point1s]

x2, y2, z2 = 4, 7, 2
m.strike_dip_rake = 32, 41, 91
point2s, triangle2s, amplitude2s = m.amplitude_distribution(4)
point2s = [point2 + [x2, y2, z2] for point2 in point2s]


points, triangles, amplitudes = eq.moment_tensor.merge_amplitude_distributions((
    (point1s, triangle1s, amplitude1s),
    (point2s, triangle2s, amplitude2s),
))
amplitudes = [math.copysign(1, a) for a in amplitudes]
print(eq.moment_tensor.vtk(points, triangles, amplitudes))
EOF

paraview meca.vtk
```

## License

GPL version 3.
