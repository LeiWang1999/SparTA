sparsity_ratio=(0.5 0.75 0.90625)
# M9 M10 M11 M12 M13 M14 M15 M16
# m 256 1024 4096 256 1024 4096 256 1024
# k 1024 1024 1024 2048 2048 2048 4096 4096
# n 1024 1024 1024 2048 2048 2048 4096 4096
# M17 M18 M19 M20 M21 M22 M23 M24
# m 4096 256 1024 4096 256 1024 4096 256
# k 4096 8192 8192 8192 1024 1024 1024 4096
# n 4096 8192 8192 8192 4096 4096 4096 1024
# M25 M26 M27 M28 M29 M30 M31 M32
# m 1024 4096 256 1024 4096 256 1024 4096
# k 4096 4096 5120 5120 5120 20480 20480 20480
# n 1024 1024 20480 20480 20480 5120 5120 5120
shape_config=(
        'M9 256 1024 1024'
        'M10 1024 1024 1024'
        'M11 4096 1024 1024'
        'M12 256 2048 2048'
        'M13 1024 2048 2048'
        'M14 4096 2048 2048'
        'M15 256 4096 4096'
        'M16 1024 4096 4096'
        'M17 4096 4096 4096'
        'M18 256 8192 8192'
        'M19 1024 8192 8192'
        'M20 4096 8192 8192'
        'M21 256 1024 4096'
        'M22 1024 1024 4096'
        'M23 4096 1024 4096'
        'M24 256 4096 1024'
        'M25 1024 4096 1024'
        'M26 4096 4096 1024'
        'M27 256 5120 20480'
        'M28 1024 5120 20480'
        'M29 4096 5120 20480'
        'M30 256 20480 5120'
        'M31 1024 20480 5120'
        'M32 4096 20480 5120'
    )


for sparsity in ${sparsity_ratio[@]}
do
    for info in "${shape_config[@]}";
    do
        name=`echo $info | awk '{print $1}'`
        m=`echo $info | awk '{print $2}'`
        k=`echo $info | awk '{print $3}'`
        n=`echo $info | awk '{print $4}'`
        echo $name $m $k $n $sparsity
        # VW64
        python ../nmsparse/run_SPMM_TC_VW64.py --sparsity_ratio $sparsity --name $name --M $m --K $k --N $n >> ./nmsparse_result.txt
        # BW64x64
        python ../nmsparse/run_SPMM_TC_BW64x64.py --sparsity_ratio $sparsity --name $name --M $m --K $k --N $n >> ./nmsparse_result.txt
    done
done
