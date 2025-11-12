# CasADiNLPModels
[![CI](https://github.com/apozharski/CasADiNLPModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/apozharski/CasADiNLPModels.jl/actions/workflows/CI.yml)

This package provides support for loading [`NLPModels`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) conforming NLPs from the compiled output of [CasADi](https://github.com/casadi/casadi) code generation.
It also provides standalone loading of arbitrary (non-NLP) functions.
For instruction on how to generate the necessary binaries, see the [CasADi documentation](https://web.casadi.org/docs/#document-ccode).

## Installation
```
pkg> add CasADiNLPModels
```
## Usage
As a prerequisite you need a compiled binary and a JSON data file, the formats of which you can find in the next section.

The required binary can be generated manually from CasADi expressions (using the Matlab interface for example):
```matlab
fname = 'nlp';
import casadi.*
nx = 10;
ng = 3;
np = 10;
% Define here the necessary symbolics:
x = SX.sym('x', nx);
p = SX.sym('x', np);
lam_f = SX.sym('lam_f', 1);
lam_g = SX.sym('x', ng);
f = sum(p.*(x.^2));
g = sprand(ng,nx,0.5,0.5)*x; % random QP with A having condition number 2 and 0.5 density
grad_f = f.gradient(x);
jac_g = g.jacobian(x);
L = lam_f*f - lam_g'*g;
[hess_L, nabla_L] = L.hessian(x);

% Generate funs
nlp_f = Function('nlp_f', {x, p}, {f}, {'x','p'}, {'f'});
nlp_grad_f = Function('nlp_grad_f', {x, p}, {f, grad_f}, {'x','p'}, {'f','grad_f'});
nlp_g = Function('nlp_g', {x, p}, {g}, {'x','p'}, {'g'});
nlp_jac_g = Function('nlp_jac_g', {x, p}, {g, jac_g}, {'x', 'p'}, {'g', 'jac_g'});
nlp_hess_l = Function('nlp_hess_l', {x, p, lam_f, lam_g}, {hess_L}, {'x', 'g', 'lam_f', 'lam_g'}, {'hess_l'});

% Generate c code
cg = CodeGenerator([fname,'.c']);
cg.add(nlp_f);
cg.add(nlp_g);
cg.add(nlp_grad_f);
cg.add(nlp_jac_g);
cg.add(nlp_hess_l);
cg.generate();

% Generate json
json_struct.x0 = zeros(nx,1);
json_struct.y0 = zeros(ng,1);
json_struct.p0 = ones(np,1);
json_struct.lbx = -ones(nx, 1);
json_struct.ubx = ones(nx, 1);
json_struct.lbg = 0.1*ones(ng,1);
json_struct.ubg = 0.1*ones(ng,1);
json = jsonencode(json_struct, "ConvertInfAndNaN", false, "PrettyPrint", true);
fid = fopen([fname, '.json'], "w");
fprintf(fid, json);
fclose(fid);
```
or by using the `generate_dependencies` method of an `nlpsol` object with the `ipopt` plugin, [see this python CasADi example for details](https://github.com/casadi/casadi/blob/main/docs/examples/python/nlp_codegen.py).
The output file called can be then compiled into a shared library using your favorite compiler, e.g. GCC:
```bash
gcc -shared -fPIC -O3 -o nlp.so nlp.c
```

Now to load the NLP Model and solve it with, for example, [MadNLP](https://github.com/MadNLP/MadNLP.jl) you can use:
```julia
using CasADiNLPModels, MadNLP
nlp = CasADiNLPModel("nlp.so", "nlp.json")
stats = madnlp(nlp)
```

## Formats
Currently the `CasADiNLPModel` constructor expects the following function names in the loaded library:
- `nlp_f`: `(x,p)->f(x;p)`. Takes the optimization variables and parameters as arguments and returns the objective.
- `nlp_g`: `(x,p)->g(x;p)`. Takes the optimization variables and parameters as arguments and returns the nonlinear constraints.
- `nlp_grad_f`: `(x,p)->(f(x;p), ∇f(x;p)`. Takes the optimization variables and parameters as arguments and returns the objective and its gradient.
- `nlp_jac_g`: `(x,p)->(g(x;p), ∇g(x;p)`. Takes the optimization variables and parameters as arguments and returns the nonlinear constraints and their Jacobian.
- `nlp_hess_l`: `(x,p,λf,λg)->(∇²L(x,λf,λg;p)`. Takes the optimization variables and parameters as arguments and returns the Hessian of the Lagrangian.


The JSON file containing the data should contain:
- `x0`: Initial point.
- `y0`: Initial multipliers.
- `p0`: Parameter values.
- `lbx`: Primal lower bounds.
- `ubx`: Primal upper bounds.
- `lbg`: Nonlinear constraint lower bounds.
- `ubg`: Nonlinear constraint upper bounds.

## Warnings
As this package loads dynamic libraries, one can, if not careful, cause strange issues, or even segfault your `julia` process.
We implement a reference counting scheme which does its best to avoid allowing the user to shoot themselves in the foot by loading a library that has changed on disk but is already loaded by the `julia` process.

## Bug Reports
Please report any bugs or issues you find using [Github issues](https://github.com/apozharski/CasADiNLPModels.jl/issues).
When reporting a bug, it would be useful to provide a minimal working example.