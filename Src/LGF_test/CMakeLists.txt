add_executable(lgfpoisson
                main.cpp
                initializeMultiFab.cpp
                solveMLMG.cpp
                residualCheck.cpp
                MyFunctions.H
                SourceField.H)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(lgfpoisson)
endif()

if(AMReX_OMP)
    target_link_libraries(lgfpoisson PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(lgfpoisson PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lgfpoisson PRIVATE LGF_core_lib AMReX::amrex)

# Direct the binary to Exec/LGF_test
set_target_properties(lgfpoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXEC_DIR}/LGF_test")

# Update the inputs file in test run folder
add_custom_command(TARGET lgfpoisson POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/inputs"
    "${EXEC_DIR}/LGF_test/inputs"
    COMMENT "Syncing inputs to Exec/LGF_test..."
)

# Update the jobscript file in test run folder
add_custom_command(TARGET lgfpoisson POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/debug.sh"
    "${EXEC_DIR}/LGF_test/debug.sh"
    COMMENT "Syncing jobscript to Exec/LGF_test..."
)

# Update the gpu jobscript file for param pravega
add_custom_command(TARGET lgfpoisson POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/multicase_gpu_pravega.sh"
    "${EXEC_DIR}/LGF_test/multicase_gpu_pravega.sh"
    COMMENT "Syncing Param Pravega GPU jobscript to Exec/LGF_test..."
)

# Update the gpu jobscript file for param pravega
add_custom_command(TARGET lgfpoisson POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/multicase_gpu_rudra.sh"
    "${EXEC_DIR}/LGF_test/multicase_gpu_rudra.sh"
    COMMENT "Syncing GPU jobscript to Exec/LGF_test..."
)

# Update the postProcess python file in test run folder
add_custom_command(TARGET lgfpoisson POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/postProcess.py"
    "${EXEC_DIR}/LGF_test/Results/postProcess.py"
    COMMENT "Syncing postprocessor to Exec/LGF_test/Results..."
)

# Update the case description file in test run folder
add_custom_command(TARGET lgfpoisson POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/description.txt"
    "${EXEC_DIR}/LGF_test/description.txt"
    COMMENT "Syncing case description file to Exec/LGF_test..."
)