source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
S=$HOME/poly-spike; R=$HOME/PolyStrata; O=$S/out
G=$R/build-sycl-aot/_deps/strata_llamacpp-src/ggml
INC="-I$O/include -I$O -I$O/src -I$R/sycl/include -I$R/include -I$R/src -I$G/include -I$G/src"
cd $S
for f in kernels_inkling kernels_dsv4 kernels; do
  ( time icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -fsycl-default-sub-group-size=32 -fp-model=precise -std=c++20 -O2 -Wno-everything -DSTRATA_VERSION=\"port\" $INC -c $O/src/poly/$f.dp.cpp -o $S/$f.o ) > aot_$f.log 2>&1
  echo "$f: exit $? errors=$(grep -c 'error' aot_$f.log) spills=$(grep -ci 'spill' aot_$f.log) size=$(stat -c %s $S/$f.o 2>/dev/null)" >> $S/aot_summary.txt
done
echo done >> $S/aot_summary.txt
