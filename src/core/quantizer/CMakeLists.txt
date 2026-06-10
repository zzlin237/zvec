include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT APPLE)
  set(CORE_QUANTIZER_LDFLAGS
      "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a")
endif()

if(RABITQ_SUPPORTED AND RABITQ_ARCH_FLAG)
  set_source_files_properties(
      record_rotator.cc
      PROPERTIES
      COMPILE_FLAGS "${RABITQ_ARCH_FLAG}"
  )
endif()

cc_library(
        NAME core_quantizer 
        STATIC SHARED STRICT ALWAYS_LINK
        SRCS *.cc
        LIBS zvec_ailego core_framework rabitqlib
        LIBS zvec_ailego zvec_turbo core_framework rabitqlib
        INCS . ${PROJECT_ROOT_DIR}/src/core
        LDFLAGS "${CORE_QUANTIZER_LDFLAGS}"
        VERSION "${PROXIMA_ZVEC_VERSION}"
)
