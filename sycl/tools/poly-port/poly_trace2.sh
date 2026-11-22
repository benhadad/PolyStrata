source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_UR_TRACE=2
M=/mnt/c/AI_Data/models/Inkling-2-Mini-downloads/GGUF/Inkling-2-Mini-UD-Q3_K_XL.gguf
cd ~/PolyStrata/build-sycl-aot
timeout 300 ./strata-poly --model $M --tokens 1,2,3,4,5 --max-new 2 --max-context 2048 --vram-reserve-mib 26000 > ~/poly-spike/trace2.log 2>&1
L=~/poly-spike/trace2.log
echo "copies: $(grep -c 'urEnqueueUSMMemcpy(.hQueue' $L)"
echo "--- device allocs (ptr,size):"
grep -E '<--- urUSMDeviceAlloc' $L | sed -E 's/.*\.size = ([0-9]+).*\.ppMem = ([^ ,)]*).*/\1 \2/' | head -40
echo "--- last 6 memcpy:"
grep 'urEnqueueUSMMemcpy(.hQueue' $L | tail -6 | sed -E 's/.*(\.blocking.*\.size = [0-9]+).*/\1/'
grep -n 'DEVICE_LOST' $L | head -2
