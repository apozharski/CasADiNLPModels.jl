struct CasADiNLPData
    x0::Vector{Cdouble}
    y0::Vector{Cdouble}
    p0::Vector{Cdouble}
    lbx::Vector{Cdouble}
    ubx::Vector{Cdouble}
    lbg::Vector{Cdouble}
    ubg::Vector{Cdouble}
end

mutable struct CasADiNLPModel <: AbstractNLPModel{Cdouble, Vector{Cdouble}}
    lib::Ptr{Cvoid}
    const f::CasADiFunction
    const grad_f::CasADiFunction
    const g::CasADiFunction
    const jac_g::CasADiFunction
    const hess_L::CasADiFunction
    const p::Vector{Cdouble}
    const np::Clong
    const meta::NLPModels.NLPModelMeta{Cdouble, Vector{Cdouble}}
    const counters::NLPModels.Counters

    function CasADiNLPModel(libpath::String, datapath::String)
        if is_lib_loaded(libpath)
            @warn "Loading NLP that is already loaded. Only the metadata will be different. If you want to load an updated NLP, make sure all referenes to CasADiNLPModels objects loading from the shared $(libpath) are unreachable."
        end
        lib = checkout_lib(libpath)
        inc_refcount(lib)
        # f(x,p)->f
        f = CasADiFunction(lib, :nlp_f)
        # grad_f(x,p)->(f, grad_f)
        grad_f = CasADiFunction(lib, :nlp_grad_f)
        # g(x,p)->g
        g = CasADiFunction(lib, :nlp_g)
        # jac_g(x,p)->(g, jac_g)
        jac_g = CasADiFunction(lib, :nlp_jac_g)
        # hess_L(x,p,lam_f, lam_g)->(hess_l)
        hess_L = CasADiFunction(lib, :nlp_hess_l)

        # Read data (for now assume that we are reading from json:
        # TODO(@anton) yell at people to update to JSONv1 :)
        #data = JSON.parsefile(datapath, CasADiNLPData; allownan=true)
        data = JSON.parsefile(datapath; allownan=true)

        # Build meta
        meta = NLPModels.NLPModelMeta{Cdouble, Vector{Cdouble}}(
            length(data["x0"]);
            name=libpath,
            ncon=length(data["lbg"]),
            x0=Vector{Cdouble}(data["x0"]),
            y0=Vector{Cdouble}(data["y0"]),
            lvar=Vector{Cdouble}(data["lbx"]),
            uvar=Vector{Cdouble}(data["ubx"]),
            lcon=Vector{Cdouble}(data["lbg"]),
            ucon=Vector{Cdouble}(data["ubg"]),
            nnzj=nnz(jac_g.out_sparsities[2]),
            nnzh=nnz(hess_L.out_sparsities[1]),
            minimize=true,
        )

        nlp = new(
            lib,
            f,
            grad_f,
            g,
            jac_g,
            hess_L,
            data["p0"],
            length(data["p0"]),
            meta,
            NLPModels.Counters(),
        )

        function free(nlp::CasADiNLPModel)
            if is_free(nlp)
                @error "Double free on CasADi NLPModel."
            end
            dec_refcount(nlp.lib)
            nlp.lib = C_NULL
        end

        finalizer(free, nlp)
        return nlp
    end
end

is_free(nlp::CasADiNLPModel) = nlp.lib == C_NULL

# Implement only old interface/what we need for MadNLP
# TODO(@anton) maybe if we release this separately we want to implement the full one
#              if only to appease the JSO folks.

function NLPModels.obj(nlp::CasADiNLPModel, x::AbstractVector{Cdouble})
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nvar x
    (f,) = nlp.f(x, nlp.p)
    return f[1]
end

function NLPModels.grad!(
    nlp::CasADiNLPModel,
    x::AbstractVector{Cdouble},
    g::AbstractVector{Cdouble},
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nvar g
    (f, grad_f) = nlp.grad_f(x, nlp.p)
    g .= grad_f
    return g
end

function NLPModels.cons!(
    nlp::CasADiNLPModel,
    x::AbstractVector{Cdouble},
    c::AbstractVector{Cdouble},
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.ncon c
    (g,) = nlp.g(x, nlp.p)
    c .= g
    return c
end

function NLPModels.objgrad!(
    nlp::CasADiNLPModel,
    x::AbstractVector{Cdouble},
    g::AbstractVector{Cdouble},
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nvar g
    (f, grad_f) = nlp.grad_f(x, nlp.p)
    g .= grad_f
    return f, g
end

# Shamelessly taken from QuadraticModels
function fill_structure!(S::SparseMatrixCSC, rows, cols)
    count = 1
    @inbounds for col in 1:size(S, 2), k in S.colptr[col]:(S.colptr[col+1]-1)
        rows[count] = S.rowval[k]
        cols[count] = col
        count += 1
    end
end

function fill_structure!(S::Matrix, rows, cols)
    (n, m) = size(S)
    count = 1
    for row in 1:n
        for col in 1:m
            rows[count] = row
            cols[count] = col
            count += 1
        end
    end
end

function fill_structure!(S::AbstractVector, rows, cols)
    (n,) = size(S)
    count = 1
    for row in 1:n
        rows[count] = row
        cols[count] = 1
        count += 1
    end
end

function fill_coord!(S::SparseMatrixCSC, vals)
    count = 1
    return vals .= S.nzval
end

function fill_coord!(S::Matrix, vals)
    (n, m) = size(S)
    count = 1
    for row in 1:n
        for col in 1:m
            vals[count] = S[row, col]
            count += 1
        end
    end
end

function fill_coord!(S::AbstractVector, vals)
    (n,) = size(S)
    count = 1
    for row in 1:n
        vals[count] = S[row]
        count += 1
    end
end

function NLPModels.jac_structure!(
    nlp::CasADiNLPModel,
    rows::AbstractVector{Clong},
    cols::AbstractVector{Clong},
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nnzj rows cols
    fill_structure!(nlp.jac_g.res_vec[2], rows, cols)
    return rows, cols
end

function NLPModels.jac_coord!(
    nlp::CasADiNLPModel,
    x::AbstractVector{Cdouble},
    vals::AbstractVector{Cdouble},
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nnzj vals
    (g, jac_g) = nlp.jac_g(x, nlp.p)
    fill_coord!(jac_g, vals)
    return vals
end

function NLPModels.hess_structure!(
    nlp::CasADiNLPModel,
    rows::AbstractVector{Clong},
    cols::AbstractVector{Clong},
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nnzh rows cols
    fill_structure!(nlp.hess_L.res_vec[1], rows, cols)
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::CasADiNLPModel,
    x::AbstractVector{Cdouble},
    y::AbstractVector{Cdouble},
    vals::AbstractVector{Cdouble};
    obj_weight=1.0,
)
    @check_free nlp "Evaluating free'd CasADiNLPModel, this should never happen."
    @lencheck nlp.meta.nnzh vals
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.ncon y
    (hess_L,) = nlp.hess_L(x, nlp.p, Vector{Cdouble}([obj_weight]), y)
    fill_coord!(hess_L, vals)
    return vals
end
