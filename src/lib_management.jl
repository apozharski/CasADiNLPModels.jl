const libdirectory = Base.OncePerProcess{String}() do
    # NOTE: This may throw if user can't create tmpdir. Oh well :)

    return mktempdir(;prefix="casadinlpmodels_")
end

macro check_free(obj, msg)
    return esc(quote
        if is_free($obj)
            @error $msg
        end
    end)
end

function generate_copypath(fname::String)
    randslug = randstring(10) # This is _probably_ enough!
    name,ext = splitext(fname)

    return joinpath(libdirectory(), name*"_"*randslug*ext)
end

function checkout_lib(libpath::String)
    abs_libpath = abspath(libpath)
    fname = splitpath(abs_libpath)[end]
    libcopypath = generate_copypath(fname)
    while isfile(libcopypath) # make sure we ignore collisions
        libcopypath = generate_copypath(fname)
    end

    cp(abs_libpath, libcopypath) # This may throw for various reasons. Oh  well :)

    return dlopen(libcopypath) # Open the copy
end
