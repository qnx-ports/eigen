# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sbronder/open_source/eigen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sbronder/open_source/eigen/build

# Include any dependencies generated for this target.
include test/CMakeFiles/allocator_cpp17.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/allocator_cpp17.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/allocator_cpp17.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/allocator_cpp17.dir/flags.make

test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o: test/CMakeFiles/allocator_cpp17.dir/flags.make
test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o: /home/sbronder/open_source/eigen/test/allocator_cpp17.cpp
test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o: test/CMakeFiles/allocator_cpp17.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sbronder/open_source/eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o"
	cd /home/sbronder/open_source/eigen/build/test && /mnt/sw/nix/store/6560mkp838syd8jpp6gdyyisczwcvs67-gcc-11.4.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o -MF CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o.d -o CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o -c /home/sbronder/open_source/eigen/test/allocator_cpp17.cpp

test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.i"
	cd /home/sbronder/open_source/eigen/build/test && /mnt/sw/nix/store/6560mkp838syd8jpp6gdyyisczwcvs67-gcc-11.4.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sbronder/open_source/eigen/test/allocator_cpp17.cpp > CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.i

test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.s"
	cd /home/sbronder/open_source/eigen/build/test && /mnt/sw/nix/store/6560mkp838syd8jpp6gdyyisczwcvs67-gcc-11.4.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sbronder/open_source/eigen/test/allocator_cpp17.cpp -o CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.s

# Object files for target allocator_cpp17
allocator_cpp17_OBJECTS = \
"CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o"

# External object files for target allocator_cpp17
allocator_cpp17_EXTERNAL_OBJECTS =

test/allocator_cpp17: test/CMakeFiles/allocator_cpp17.dir/allocator_cpp17.cpp.o
test/allocator_cpp17: test/CMakeFiles/allocator_cpp17.dir/build.make
test/allocator_cpp17: test/CMakeFiles/allocator_cpp17.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sbronder/open_source/eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable allocator_cpp17"
	cd /home/sbronder/open_source/eigen/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/allocator_cpp17.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/allocator_cpp17.dir/build: test/allocator_cpp17
.PHONY : test/CMakeFiles/allocator_cpp17.dir/build

test/CMakeFiles/allocator_cpp17.dir/clean:
	cd /home/sbronder/open_source/eigen/build/test && $(CMAKE_COMMAND) -P CMakeFiles/allocator_cpp17.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/allocator_cpp17.dir/clean

test/CMakeFiles/allocator_cpp17.dir/depend:
	cd /home/sbronder/open_source/eigen/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sbronder/open_source/eigen /home/sbronder/open_source/eigen/test /home/sbronder/open_source/eigen/build /home/sbronder/open_source/eigen/build/test /home/sbronder/open_source/eigen/build/test/CMakeFiles/allocator_cpp17.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/allocator_cpp17.dir/depend

