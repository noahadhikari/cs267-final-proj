cmake_minimum_required(VERSION 3.14)
project(final-proj)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(sparse main.cpp common.h)
target_link_libraries(sparse PRIVATE MPI::MPI_CXX)

# Copy the job scripts
configure_file(job-sparse job-sparse COPYONLY)
