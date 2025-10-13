from qibo import set_backend, set_dtype

set_backend("numpy")  # enables the numpy backend
set_dtype("complex64") # enables complex64

# alternatively, it is possible to set backend and data type in the same line.
# The following line re-enables the numpy backend but now with complex128 data type.
set_backend("numpy", dtype="complex128")

# ... continue with circuit creation and execution