from importlib.util import find_spec

CPP_EXT_AVAILABLE = find_spec(".cpp_algo", package="cayleypy") is not None
