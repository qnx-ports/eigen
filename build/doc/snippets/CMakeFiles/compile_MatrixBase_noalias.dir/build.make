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
include doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compiler_depend.make

# Include the progress variables for this target.
include doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/progress.make

# Include the compile flags for this target's objects.
include doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/flags.make

doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o: doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/flags.make
doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o: doc/snippets/compile_MatrixBase_noalias.cpp
doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o: /home/sbronder/open_source/eigen/doc/snippets/MatrixBase_noalias.cpp
doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o: doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sbronder/open_source/eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o"
	cd /home/sbronder/open_source/eigen/build/doc/snippets && /mnt/sw/nix/store/6560mkp838syd8jpp6gdyyisczwcvs67-gcc-11.4.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o -MF CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o.d -o CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o -c /home/sbronder/open_source/eigen/build/doc/snippets/compile_MatrixBase_noalias.cpp

doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.i"
	cd /home/sbronder/open_source/eigen/build/doc/snippets && /mnt/sw/nix/store/6560mkp838syd8jpp6gdyyisczwcvs67-gcc-11.4.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sbronder/open_source/eigen/build/doc/snippets/compile_MatrixBase_noalias.cpp > CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.i

doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.s"
	cd /home/sbronder/open_source/eigen/build/doc/snippets && /mnt/sw/nix/store/6560mkp838syd8jpp6gdyyisczwcvs67-gcc-11.4.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sbronder/open_source/eigen/build/doc/snippets/compile_MatrixBase_noalias.cpp -o CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.s

# Object files for target compile_MatrixBase_noalias
compile_MatrixBase_noalias_OBJECTS = \
"CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o"

# External object files for target compile_MatrixBase_noalias
compile_MatrixBase_noalias_EXTERNAL_OBJECTS =

doc/snippets/compile_MatrixBase_noalias: doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/compile_MatrixBase_noalias.cpp.o
doc/snippets/compile_MatrixBase_noalias: doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/build.make
doc/snippets/compile_MatrixBase_noalias: doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sbronder/open_source/eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compile_MatrixBase_noalias"
	cd /home/sbronder/open_source/eigen/build/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_MatrixBase_noalias.dir/link.txt --verbose=$(VERBOSE)
	cd /home/sbronder/open_source/eigen/build/doc/snippets && ./compile_MatrixBase_noalias >/home/sbronder/open_source/eigen/build/doc/snippets/MatrixBase_noalias.out

# Rule to build all files generated by this target.
doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/build: doc/snippets/compile_MatrixBase_noalias
.PHONY : doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/build

doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/clean:
	cd /home/sbronder/open_source/eigen/build/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_MatrixBase_noalias.dir/cmake_clean.cmake
.PHONY : doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/clean

doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/depend:
	cd /home/sbronder/open_source/eigen/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sbronder/open_source/eigen /home/sbronder/open_source/eigen/doc/snippets /home/sbronder/open_source/eigen/build /home/sbronder/open_source/eigen/build/doc/snippets /home/sbronder/open_source/eigen/build/doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/snippets/CMakeFiles/compile_MatrixBase_noalias.dir/depend

