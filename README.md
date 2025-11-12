# CasADiNLPModels
This package provides support for loading `NLPModels` conforming NLPs from the compiled output of (CasADi)[https://github.com/casadi/casadi] code generation.
It also provides standalone loading of arbitrary (non-NLP) functions.
For instruction on how to generate the necessary binaries, see the (CasADi documentation)[https://web.casadi.org/docs/#document-ccode].

## Usage
As a prerequisite you need a compiled binary and a JSON data file, the formats of which you can find in the next section.
To generate an NLP Model and solve it with, for example, (MadNLP)[https://github.com/MadNLP/MadNLP.jl] you can use:
```julia
using CasADiNLPModels, MadNLP
nlp = CasADiNLPModel('nlp.so', 'nlp.json')
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
- `lbx`: Primal lower bounds.
- `ubx`: Primal upper bounds.
- `lbg`: Nonlinear constraint lower bounds.
- `ubg`: Nonlinear constraint upper bounds.

## Warnings
As this package loads dynamic libraries, one can, if not careful, cause strange issues, or even segfault your `julia` process.
We implement a refcounting scheme which does it's best to avoid allowing the user to shoot themselves in the foot by loading a library that has changed on disk but is already loaded by the `julia` process.