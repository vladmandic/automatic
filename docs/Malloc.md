# Memory Allocator

Combination of Linux default memory allocator `malloc` with Python's default memory allocator is pessimistic when it comes to system memory garbage collection and it will sometimes hold on to allocated memory longer than necessary even if GC is triggered explicitly.  
This appears to user as a memory leak as process memory usage grows over time.  
This is especially noticeable when frequently loading/unloading large objects such as models or LoRAs.  

!!! tip

    For Linux deployments you can switch out memory allocator to `tcmalloc` or `jemalloc`  
    which are more efficient and have better memory management.

!!! note

    This applies to system memory only and has no impact on GPU memory management

> tcmalloc

    sudo apt install google-perftools  
    sudo ldconfig  
    export LD_PRELOAD=libtcmalloc.so.4  
    ./webui.sh --debug  

> jemalloc

    sudo apt install libjemalloc2
    sudo ldconfig  
    export LD_PRELOAD=libjemalloc.so.2  
    ./webui.sh --debug
