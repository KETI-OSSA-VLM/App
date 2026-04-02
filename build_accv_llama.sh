#!/bin/bash
/home/keti/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++ \
  -shared -fPIC -march=armv8.2-a+dotprod -O2 -std=c++17 \
  -I/mnt/c/android/accv/external/llama.cpp/include \
  -I/mnt/c/android/accv/external/llama.cpp/ggml/include \
  -I/mnt/c/android/accv/external/llama.cpp/tools/mtmd \
  -I/mnt/c/android/accv/external/llama.cpp \
  -L/mnt/c/android/accv/app/src/main/jniLibs/arm64-v8a \
  -lllama -lmtmd -landroid -llog \
  /mnt/c/android/accv/app/src/main/cpp/llama_bridge.cpp \
  -o /tmp/libaccv_llama.so

if [ $? -eq 0 ]; then
  cp /tmp/libaccv_llama.so /mnt/c/android/accv/app/src/main/jniLibs/arm64-v8a/libaccv_llama.so
  echo "Done! libaccv_llama.so copied to jniLibs."
else
  echo "Build failed."
fi
