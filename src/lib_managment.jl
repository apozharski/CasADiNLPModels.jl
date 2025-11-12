const lib_refcount = Dict{Ptr{Nothing}, Tuple{String, Int}}()
const lib_paths = Dict{String, Ptr{Nothing}}()

macro check_free(obj, msg)
    return quote
        if is_free($obj)
            @error $msg
        end
    end
end

function is_lib_loaded(libpath::String)
    GC.gc() # Make sure any unreachable CasADiNLPModels objects are cleaned up
    haskey(lib_paths, libpath)
end

function checkout_lib(libpath::String)
    GC.gc() # Make sure any unreachable CasADiNLPModels objects are cleaned up
    if haskey(lib_paths, libpath)
        lib = lib_paths[libpath]
        return lib
    else
        lib = Libdl.dlopen(libpath)
        lib_refcount[lib] = (libpath, 0)
        lib_paths[libpath] = lib
        return lib
    end
end

function inc_refcount(lib::Ptr{Nothing})
    GC.gc() # Make sure any unreachable CasADiNLPModels objects are cleaned up
    if haskey(lib_refcount, lib)
        (path, count) = lib_refcount[lib]
        count += 1
        # increase the library refcount
        lib_refcount[lib] = (path, count)
    else
        @warn "inc_refcount did nothing because the passed pointer is not in the refcount dictionary. This means something is extremely wrong."
    end
end

function dec_refcount(lib::Ptr{Nothing})
    GC.gc() # Make sure any unreachable CasADiNLPModels objects are cleaned up
    if haskey(lib_refcount, lib)
        (path, count) = lib_refcount[lib]
        count -= 1
        if count < 0
            @error "CasADiNLPModels refcounter is in irrecoverable state. This should never happen, please contact the developers."
        elseif count == 0
            # Free the library
            delete!(lib_refcount, lib)
            delete!(lib_paths, path)
            Libdl.dlclose(lib)
        else
            # Decrease the library refcount
            lib_refcount[lib] = (path, count)
        end
    else
        @warn "dec_refcount did nothing because the passed pointer is not in the refcount dictionary. This means something is extremely wrong."
    end
end
