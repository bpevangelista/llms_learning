filename="$1"
nvcc -O2 -o ${filename%.*} $1 -I../../../../cutlass/include --gpu-architecture sm_80 --expt-relaxed-constexpr
