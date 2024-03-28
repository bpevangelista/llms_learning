filename="$1"
# release build
nvcc -O2 -o ${filename%.*} $1 -I../../../../cutlass/include --gpu-architecture sm_80 --expt-relaxed-constexpr

# debug build
#nvcc -g -O0 -DDEBUG -o ${filename%.*} $1 -I../../../../cutlass/include --gpu-architecture sm_80 --expt-relaxed-constexpr
