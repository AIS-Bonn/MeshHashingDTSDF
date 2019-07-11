# Try to find Cap'n'P

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(CAPNP QUIET capnp)
  pkg_check_modules(KJ QUIET capnp)
endif()
