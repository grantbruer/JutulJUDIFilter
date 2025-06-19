using .CUDA
using Distributed

function distribute_gpus()
    if ! CUDA.functional()
        println("No gpu")
        return
    end
    @show collect(devices())
    @show deviceid.(devices())
    @show nprocs()
    @show nworkers()
    @show ENV["SLURM_JOB_GPUS"]

    # Assign GPUs
    N = length(devices())
    n_workers_per_gpu = floor(Int, nworkers() / N)
    n_extra = nworkers() % N
    start = 1
    for (i, d) in enumerate(devices())
        finish = start + n_workers_per_gpu - 1
        if i <= n_extra
            finish = finish + 1
        end
        @show i d start finish
        for p = start:finish
            remotecall_wait(p) do
                @info "Worker $p uses $d"
                device!(d)
            end
        end
        start = finish + 1
    end
end
