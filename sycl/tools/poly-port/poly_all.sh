#!/bin/bash
# Poly engine -> SYCL: dpct + fixups + syntax check.  Scratch tree: ~/poly-spike/out
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
W=/mnt/h/dev/polystrata-work
R=$HOME/PolyStrata; S=$HOME/poly-spike; O=$S/out
G=$R/build-sycl-aot/_deps/strata_llamacpp-src/ggml
rm -rf $S; mkdir -p $S
cd $R
export CUDA_HEADERS=$HOME/cuda-headers/include
export MIGRATE_ONLY="src/poly/kernels.cu src/poly/kernels_dsv4.cu src/poly/kernels_inkling.cu src/poly/engine.cpp src/poly/inkling.cpp src/poly/spill.cpp src/poly/main.cpp"
export DPCT_EXTRA="--extra-arg=-I$G/include --extra-arg=-I$G/src"
bash sycl/tools/migrate.sh $R $O > $S/migrate.stdout 2>&1; echo "migrate exit $?"
mkdir -p $O/tools && cp $R/sycl/tools/fixups.py $O/tools/
python3 $O/tools/fixups.py > $S/fixups.log 2>&1; echo "repo fixups: $(grep -c 'fixed' $S/fixups.log) files"
python3 $W/ink_fixups.py $O > $S/ink_fixups.log 2>&1; tail -2 $S/ink_fixups.log
python3 $W/poly_fixups.py $O 2>&1 | tail -6
INC="-I$O/include -I$O -I$O/src -I$R/sycl/include -I$R/include -I$R/src -I$G/include -I$G/src"
cd $S
for f in src/poly/kernels.dp.cpp src/poly/kernels_dsv4.dp.cpp src/poly/kernels_inkling.dp.cpp src/poly/engine.cpp.dp.cpp src/poly/inkling.cpp.dp.cpp; do
  b=$(basename $f)
  icpx -fsycl -std=c++20 -fsyntax-only -ferror-limit=0 -Wno-everything -DSTRATA_VERSION=\"port\" $INC $O/$f > syn_$b.log 2>&1
  echo "$b: exit $? errors=$(grep -c 'error:' syn_$b.log)"
done
grep -h -A2 'error:' syn_*.log | cut -c1-230 | head -30
