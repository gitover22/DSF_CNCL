# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zouguoqiang/cncl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zouguoqiang/cncl/build

# Include any dependencies generated for this target.
include CMakeFiles/all2all_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/all2all_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/all2all_demo.dir/flags.make

CMakeFiles/all2all_demo.dir/all2all_demo.cc.o: CMakeFiles/all2all_demo.dir/flags.make
CMakeFiles/all2all_demo.dir/all2all_demo.cc.o: ../all2all_demo.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zouguoqiang/cncl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/all2all_demo.dir/all2all_demo.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/all2all_demo.dir/all2all_demo.cc.o -c /home/zouguoqiang/cncl/all2all_demo.cc

CMakeFiles/all2all_demo.dir/all2all_demo.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/all2all_demo.dir/all2all_demo.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zouguoqiang/cncl/all2all_demo.cc > CMakeFiles/all2all_demo.dir/all2all_demo.cc.i

CMakeFiles/all2all_demo.dir/all2all_demo.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/all2all_demo.dir/all2all_demo.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zouguoqiang/cncl/all2all_demo.cc -o CMakeFiles/all2all_demo.dir/all2all_demo.cc.s

# Object files for target all2all_demo
all2all_demo_OBJECTS = \
"CMakeFiles/all2all_demo.dir/all2all_demo.cc.o"

# External object files for target all2all_demo
all2all_demo_EXTERNAL_OBJECTS =

all2all_demo: CMakeFiles/all2all_demo.dir/all2all_demo.cc.o
all2all_demo: CMakeFiles/all2all_demo.dir/build.make
all2all_demo: CMakeFiles/all2all_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zouguoqiang/cncl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable all2all_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/all2all_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/all2all_demo.dir/build: all2all_demo

.PHONY : CMakeFiles/all2all_demo.dir/build

CMakeFiles/all2all_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/all2all_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/all2all_demo.dir/clean

CMakeFiles/all2all_demo.dir/depend:
	cd /home/zouguoqiang/cncl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zouguoqiang/cncl /home/zouguoqiang/cncl /home/zouguoqiang/cncl/build /home/zouguoqiang/cncl/build /home/zouguoqiang/cncl/build/CMakeFiles/all2all_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/all2all_demo.dir/depend

