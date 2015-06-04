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

### Output a focal mechanism in VTK format for 3D visualization

```bash
python3 <<EOF > meca.vtk
import eq.moment_tensor

m = eq.moment_tensor.MomentTensor()
m.strike_dip_rake = 42, 80, 15
p, t, a = m.amplitude_distribution(7)
print(eq.moment_tensor.vtk(p, t, a))
EOF

paraview meca.vtk
```

## License

GPL version 3.
