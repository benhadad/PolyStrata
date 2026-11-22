source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
M=/mnt/c/AI_Data/models/Inkling-2-Mini-downloads/GGUF/Inkling-2-Mini-UD-Q3_K_XL.gguf
cd ~/PolyStrata/build-sycl-aot
timeout 900 stdbuf -oL ./strata-poly --model $M --tokens "$1" --max-new ${2:-16} --max-context 2048 ${3} > ~/poly-spike/run2.log 2>&1
echo "RUN DONE $?" >> ~/poly-spike/run2.log
cat ~/poly-spike/run2.log | cut -c1-250
