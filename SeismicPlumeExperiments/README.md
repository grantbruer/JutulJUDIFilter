# SeismicPlumeExperiments

Here I have code for testing different state-estimating algorithms using seismic data.

## Environments

I currently use Julia 1.8.5.

`envs` has the Julia environment specifications that I use for this project.
- SeismicPlume: runs the transition and observation operators
- EnKF: does the EnKF data assimilation on observations
- JustObservations: used for least squares inversion on observations
- NoObservations: does nothing with observations
- NormalizingFlow: does conditional normalizing with observations


## Parameters

`params.toml` has parameters that can be set to control the simulations and filters.

I would like to document the parameters here, but I reluctantly will not do that yet. To ensure correctness, for now, it is better to force the user either to not change a parameter or figure out exactly what it does.

Consider using `grep -Hrn` to find things.

Most parameters are used in few locations and have a very localized effect.
For instance, I can find that the CO2 viscosity parameter is read in exactly one spot.
```console
$ grep -Hrn '"viscosity_CO2"' lib scripts
./lib/params.jl:239:        :visCO2 => params["transition"]["viscosity_CO2"],
```

Some parameters will be more difficult to track down and you can just ask me what they do.
```console
$ grep -Hrn '"algorithm"' lib scripts
lib/params.jl:270:    algorithm = params["filter"]["algorithm"]
lib/params.jl:284:    algorithm = params["filter"]["algorithm"]
lib/params.jl:298:    algorithm = params["filter"]["algorithm"]
```

## Local workflow

### Initialization

0. Make a directory for storing your data.
    - `$ mkdir run`
1. Set parameters to desired values. I will just use `params/base.toml`.
2. Generate ground-truth plume data and seismic data.
    - `$ julia --project=. scripts/ground_truth_plume.jl params/base.toml run`
    - `$ julia --project=. scripts/ground_truth_seismic.jl params/base.toml run`
    - This creates files:
        - `run/ground_truth_plume_time0.jld2` for the initial time step fluid data.
        - `run/ground_truth_plume_timeX.jld2` for the later time steps.
        - `run/ground_truth_seismic_time0.jld2` for the initial time step seismic data.
        - `run/ground_truth_seismic_timeX.jld2` for later time steps.
3. Make plots.
    - `$ julia --project=. scripts/ground_truth_figures.jl params/base.toml run`
4. Initialize the filter. For example, the ensemble Kalman filter creates an initial ensemble.
    - `$ julia --project=. scripts/filter_initialize.jl params/base.toml run`
    - This creates files:
        - `run/enkf_N256/filter_0_posterior` for ensemble metadata.
        - `run/enkf_N256/filter_0_posterior_ensemble/X.jld2` for each ensemble member.


### Assimilation loop

Let `k` be the time step for which we'd like to assimilate data. We'll start with `k=1`.

1. First, transition the distribution to the next time step.
    - `$ julia --project=. scripts/filter_transition.jl params/base.toml run $k closer`
    - This creates files:
        - `run/enkf_N256/filter_${k}_prior` for ensemble metadata.
        - `run/enkf_N256/filter_${k}_prior_ensemble/X.jld2` for each ensemble member.
2. Then, simulate observations at this time step.
    - `$ julia --project=. scripts/filter_observe.jl params/base.toml run $k closer`
    - This creates files:
        - `run/enkf_N256/filter_obs_${k}_prior` for ensemble metadata.
        - `run/enkf_N256/filter_obs_${k}_prior_ensemble/X.jld2` for each ensemble member.
3. Then, assimilate observation of the ground-truth at this time step.
    - `$ julia --project=. scripts/filter_assimilate.jl params/base.toml run $k`
    - This creates files:
        - `run/enkf_N256/filter_${k}_state_update` for ensemble metadata.
        - `run/enkf_N256/filter_${k}_state_update_ensemble/X.jld2` for each ensemble member.
4. Then, process the assimilated states to get them ready for the two-phase flow simulator.
    - `$ julia --project=. scripts/filter_process_assimilation.jl params/base.toml run $k`
    - This creates files:
        - `run/enkf_N256/filter_${k}_posterior` for ensemble metadata.
        - `run/enkf_N256/filter_${k}_posterior_ensemble/X.jld2` for each ensemble member.

Then repeat with `k=2`, and so on.


## Slurm workflow

### Create ground-truth data.

1. `params/base.toml` has the parameters I use for the ground truth.

2. I create soft-links to the `/slimdata` partition where I store my big files.

```console
$ ln -s /slimdata/gbruer3/JutulJUDIFilter/compass compass
$ ls -lh compass/
total 2.3G
-rw-r--r-- 1 gbruer3 coc-cruyff-access 6.8M Nov 29 16:13  BGCompass_tti_625m.jld2
-rw-r--r-- 1 gbruer3 coc-cruyff-access 2.8K Nov 29 16:18  Compass256_idx.jld2
-rw-r--r-- 1 gbruer3 coc-cruyff-access 2.5G Nov 29 16:12 'broad&narrow_perm_models_new.jld2'
$ mkdir /slimdata/gbruer3/JutulJUDIFilter/run_v4
$ ln -s /slimdata/gbruer3/JutulJUDIFilter/run_v4 run
$ ls -og
total 54
lrwxrwxrwx 1    42 Feb  7 11:01 compass -> /slimdata/gbruer3/JutulJUDIFilter/compass/
drwxr-xr-x 2     3 Mar  4 15:46 docs
drwxr-xr-x 9     9 Mar  7 15:29 envs
drwxr-xr-x 2    27 Mar  7 15:33 job_scripts
drwxr-xr-x 2     4 Mar  7 10:26 lib
drwxr-xr-x 2     6 Mar  4 15:46 oneoffs
drwxr-xr-x 2    23 Mar  7 17:04 params
-rw-r--r-- 1 14296 Mar  4 15:46 README.md
lrwxrwxrwx 1    43 Feb  7 11:38 run -> /slimdata/gbruer3/JutulJUDIFilter/run_v4
drwxr-xr-x 2    16 Mar  7 15:51 scripts
```

3. I double-check the job script files I'm about to run to make sure I am requesting the number of resources I want.
    - `job_scripts/*/ground_truth_plume.sh` generates the plume data.
    - `job_scripts/*/ground_truth_seismic.sh` generates the seismic data.
    - `job_scripts/*/ground_truth_figures.sh` generates the figures.

4. Generate the plumes and make sure they look right.

```console
$ ./job_scripts/cruyff/ground_truth_plume.sh params/base.toml run
12123
$ ./job_scripts/cruyff/ground_truth_figures.sh params/base.toml run afterok:12123
12124
```

5. Figure out why the first job errored.

```console
$ squeue -u gbruer3
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             12124       cpu   fig-gt  gbruer3 PD       0:00      1 (DependencyNeverSatisfied)
             12120       cpu interact  gbruer3  R      56:46      1 epyc0
$ vim run/slurm/slurm-out-plume-gt-12123.txt
```

6. Oh, it's because I submitted from an interactive session. I'll try again from the login node and update the dependency definition for the figures job.

```console
$ scancel 12124
$ ./job_scripts/cruyff/ground_truth_plume.sh params/base.toml run
12126
$ ./job_scripts/cruyff/ground_truth_figures.sh params/base.toml run afterok:12126
12127
```

7. I don't want to sit here waiting, so I'm gonna try to get an email when the figures ar e ready.

```console
$ scontrol update jobid=12127 MailType=END MailUser='redacted@grantmail.com'
```

8. The job finished so I'm gonna `rsync` the figures to my laptop for viewing. (Note: the email thing did not work.)

```console
$ mkdir run_v4_local && cd run_v4_local
$ rsync -avxS cruyff:/slimdata/gbruer3/JutulJUDIFilter/run_v4/figs .
$ xdg-open figs
```

9. The figures look good, so I'll run the seismic observations.

```console
$ ./job_scripts/cruyff/ground_truth_seismic.sh params/base.toml run
12128
$ ./job_scripts/cruyff/ground_truth_figures.sh params/base.toml run afterok:12128
12130
```

10. Whenever those jobs finish, I'll look at the figures. But NoObs doesn't use observations, so I can go ahead and get that going.


### Fine-grained error-prone run with NoObs filter

I've made a lot of changes to the code, so I'm going to take very small steps here to make it easier to debug and not hog compute resources on too many failing jobs.

1. Check `job_scripts/cruyff/filter_initialize.sh` and make sure it looks right.

2. Initialize the filter.

```console
$  ./job_scripts/cruyff/filter_initialize.sh params/noobs-base.toml run
12131
```

3. Check `job_scripts/cruyff/filter_transition.sh` and make sure it looks right.

4. Check `job_scripts/cruyff/filter_transition_array.sh` and make sure it looks right.

5. Test transitioning a couple ensemble members to time step 1.

```console
$ ./job_scripts/cruyff/filter_transition.sh
usage: ./job_scripts/cruyff/filter_transition.sh params_file job_dir step_index worker_type [dependency_string]

The dependency string should look something like afterok:7475, but you could probably do
 command injection with it.
$ ./job_scripts/cruyff/filter_transition.sh params/noobs-base.toml run 1 helper-1-128 afterok:12131
12132
```

6. No errors so far, so I'll go ahead and transition all the ensemble members to time step 1. In my experience, Jutul will start getting slow after doing about 20 ensemble members, so I'll use 16 jobs plus one at the end to wrap everything up.

```console
$ ./job_scripts/cruyff/filter_transition_array.sh
usage: ./job_scripts/cruyff/filter_transition_array.sh params_file job_dir step_index worker_type num_jobs [dependency_string]

The dependency string should look something like afterok:7475, but you could probably do
 command injection with it.
$ ./job_scripts/cruyff/filter_transition_array.sh params/noobs-base.toml run 1 helper 16
12134
$ /job_scripts/cruyff/filter_transition_array.sh params/noobs-base.toml run 1 closer 1 afterok:12134
12137
```

7. These seem to be running fine, and not many other jobs are running, so I'll let a few more run at once.

```console
$ scontrol update jobid=12134 ArrayTaskThrottle=4
$ squeue -u gbruer3
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    12134_[5-16%4]       cpu   trans1  gbruer3 PD       0:00      1 (JobArrayTaskLimit)
       12137_[1%2]       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
           12134_3       cpu   trans1  gbruer3  R       0:04      1 epyc0
           12134_4       cpu   trans1  gbruer3  R       0:04      1 epyc0
           12134_1       cpu   trans1  gbruer3  R       9:01      1 epyc0
           12134_2       cpu   trans1  gbruer3  R       9:01      1 epyc0

```

8. And I want to go to sleep, so I'll go ahead and queue up the rest of the jobs for this time step. NoObs doesn't need observations, so I'll make sure to not use any GPUs in these next jobs.

```console
$ ./job_scripts/cruyff/filter_observe.sh
usage: ./job_scripts/cruyff/filter_observe.sh params_file job_dir step_index worker_type [dependency_string]

The dependency string should look something like afterok:7475, but you could probably do
 command injection with it.
$ ./job_scripts/cruyff/filter_observe.sh params/noobs-base.toml run 1 closer afterok:12137
12140
$ ./job_scripts/cruyff/filter_assimilate.sh
usage: ./job_scripts/cruyff/filter_assimilate.sh params_file job_dir step_index [dependency_string]

The dependency string should look something like afterok:7475, but you could probably do
 command injection with it.
$ ./job_scripts/cruyff/filter_assimilate.sh params/noobs-base.toml run 1 afterok:12140
12141
```

9. The assimilate job failed, so I fixed several issues and requeued it.

```console
$ ./job_scripts/cruyff/filter_assimilate.sh params/noobs-base.toml run 1
12185
$ sleep 120 && ls -l run/noobs_N256/
total 174
drwxr-sr-x 2 gbruer3 coc-cruyff-access  258 Feb 10 10:57 filter_1_state_update_ensemble
-rw-r--r-- 1 gbruer3 coc-cruyff-access 5368 Feb 10 10:57 filter_1_state_update
-rw-r--r-- 1 gbruer3 coc-cruyff-access 5368 Feb 10 01:40 filter_obs_1_prior
drwxr-sr-x 2 gbruer3 coc-cruyff-access  258 Feb 10 01:40 filter_obs_1_prior_ensemble
-rw-r--r-- 1 gbruer3 coc-cruyff-access 5368 Feb 10 01:32 filter_1_prior
drwxr-sr-x 2 gbruer3 coc-cruyff-access  258 Feb 10 01:23 filter_1_prior_ensemble
drwxr-sr-x 2 gbruer3 coc-cruyff-access  258 Feb  9 23:21 filter_0_posterior_ensemble
-rw-r--r-- 1 gbruer3 coc-cruyff-access 5368 Feb  9 23:21 filter_0_posterior
````

10. The assimilate job looks fine, so I'm gonna queue up the next time step and make figures to make sure everything's fine.

```console
$ ./job_scripts/queue_full_step.sh
usage: ./job_scripts/queue_full_step.sh params_file job_dir step_index num_trans_jobs num_obs_jobs run_dependency
  the run_dependency can be 'nothing' or something like 'afterok:8744'
$ ./job_scripts/queue_full_step.sh params/noobs-base.toml run 2 16 0 nothing
12189
```

11. The assimilate job had an issue so I put the second assimilate job on hold while I checked on it.

```console
$ scontrol hold 12189
```

12. I'm ready to test the assimilate job, so I'm gonna release the NoObs one. I also put the NF one on hold, so I'm gonna double-check the job ids to make sure I don't release the NF one.

```console
$ scontrol write batch_script 12189 - | grep params
/usr/bin/time --verbose srun julia scripts/filter_assimilate.jl "params/noobs-base.toml" "run" "2"
$ scontrol write batch_script 12197 - | grep params
/usr/bin/time --verbose srun julia scripts/filter_assimilate.jl "params/nf-base.toml" "run" "1"
$ scontrol release 12189
```

13. The assimilate ran fine, so I'm queuing up the rest of the time steps.

```console
$ ./job_scripts/queue_multiple_steps.sh params/noobs-base.toml run 3 5 16 0 nothing
12310
```

### Coarser-grained error fixing with EnKF filter

The NoObs seems to be working fine, so I'll start using fancier job scripts.

1. Double check that `job_scripts/queue_full_step.sh` looks right.

2. Queue up time step 0, which just initializes the filter ensemble.

```console
$ ./job_scripts/queue_full_step.sh
usage: ./job_scripts/queue_full_step.sh params_file job_dir step_index num_trans_jobs num_obs_jobs run_dependency
  the run_dependency can be 'nothing' or something like 'afterok:8744'
$ ./job_scripts/queue_full_step.sh params/enkf-base.toml run 0 16 8 nothing
12146
```

3. The initialization probably won't fail, so go ahead and queue up the next time step.

```console
$ ./job_scripts/queue_full_step.sh params/enkf-base.toml run 1 16 8 afterok:12146
12151
```

4. Uh oh, I just double-checked `job_scripts/cruyff/filter_observe_array.sh` and found it would load the wrong Julia version. Better fix it before that job array runs.

```console
$ squeue -u gbruer3
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    12134_[9-16%4]       cpu   trans1  gbruer3 PD       0:00      1 (JobArrayTaskLimit)
       12137_[1%2]       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12140       cpu     obs1  gbruer3 PD       0:00      1 (Dependency)
             12141       cpu assimila  gbruer3 PD       0:00      1 (Dependency)
    12147_[1-16%2]       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12148       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12151       cpu assimila  gbruer3 PD       0:00      1 (Dependency)
           12134_8       cpu   trans1  gbruer3  R       5:19      1 epyc0
           12134_7       cpu   trans1  gbruer3  R       5:52      1 epyc0
           12134_6       cpu   trans1  gbruer3  R      15:06      1 epyc0
           12134_5       cpu   trans1  gbruer3  R      16:32      1 epyc0
             12146       cpu initiali  gbruer3  R       2:14      1 epyc0
     12149_[1-8%2]       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
             12150       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
$ scancel 12149
$ vim job_scripts/cruyff/filter_observe_array.sh # Fix Julia module version.
$ ./job_scripts/cruyff/filter_observe_array.sh params/enkf-base.toml run 1 helper 8 afterok:12147
12154

```

5. Uh oh, the closer job for EnKF observations is running early since the helper jobs finished (by which I mean, I killed them). So I've got to kill it and put it in the queue again in the right spot.

```console
$ scancel 12150
$ squeue -u gbruer3
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
   12134_[10-16%4]       cpu   trans1  gbruer3 PD       0:00      1 (JobArrayTaskLimit)
       12137_[1%2]       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12140       cpu     obs1  gbruer3 PD       0:00      1 (Dependency)
             12141       cpu assimila  gbruer3 PD       0:00      1 (Dependency)
    12147_[3-16%2]       cpu   trans1  gbruer3 PD       0:00      1 (JobArrayTaskLimit)
             12148       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12151       cpu assimila  gbruer3 PD       0:00      1 (DependencyNeverSatisfied)
           12134_9       cpu   trans1  gbruer3  R       0:57      1 epyc0
           12134_8       cpu   trans1  gbruer3  R      13:26      1 epyc0
           12134_7       cpu   trans1  gbruer3  R      13:59      1 epyc0
           12134_5       cpu   trans1  gbruer3  R      24:39      1 epyc0
           12147_1       cpu   trans1  gbruer3  R       6:02      1 epyc0
           12147_2       cpu   trans1  gbruer3  R       6:02      1 epyc0
     12154_[1-8%2]       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
$ ./job_scripts/cruyff/filter_observe.sh
usage: ./job_scripts/cruyff/filter_observe.sh params_file job_dir step_index worker_type [dependency_string]

The dependency string should look something like afterok:7475, but you could probably do
 command injection with it.
$ ./job_scripts/cruyff/filter_observe.sh params/enkf-base.toml run 1 closer afterany:12154
12156
$ scontrol update jobid=12151 dependency=afterok:12156
$ squeue -u gbruer3
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
   12134_[10-16%4]       cpu   trans1  gbruer3 PD       0:00      1 (JobArrayTaskLimit)
       12137_[1%2]       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12140       cpu     obs1  gbruer3 PD       0:00      1 (Dependency)
             12141       cpu assimila  gbruer3 PD       0:00      1 (Dependency)
    12147_[3-16%2]       cpu   trans1  gbruer3 PD       0:00      1 (JobArrayTaskLimit)
             12148       cpu   trans1  gbruer3 PD       0:00      1 (Dependency)
             12151       cpu assimila  gbruer3 PD       0:00      1 (Dependency)
           12134_9       cpu   trans1  gbruer3  R       4:41      1 epyc0
           12134_8       cpu   trans1  gbruer3  R      17:10      1 epyc0
           12134_7       cpu   trans1  gbruer3  R      17:43      1 epyc0
           12134_5       cpu   trans1  gbruer3  R      28:23      1 epyc0
           12147_1       cpu   trans1  gbruer3  R       9:46      1 epyc0
           12147_2       cpu   trans1  gbruer3  R       9:46      1 epyc0
     12154_[1-8%2]       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
             12156       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
```

### Coarse-grained no-errors example with NormFlow filter

The `job_scripts/queue_full_step.sh` seems to be working fine, so I'll try out the fancier `job_scripts/queue_multi_step.sh` now.

**Important:** The NormFlow filter uses the GPU during the assimilation step, so I have to make sure to use the job script that requests the GPU.

1. Queue up the first two steps.

```console
$ ./job_scripts/queue_multiple_steps_assim-gpu.sh
usage: ./job_scripts/queue_multiple_steps_assim-gpu.sh params_file job_dir step_index_start step_index_end num_trans_jobs num_obs_jobs run_dependency
  the run_dependency can be 'nothing' or something like 'afterok:8744'
$ ./job_scripts/queue_multiple_steps_assim-gpu.sh params/nf-base.toml run 0 1 16 8 nothing
12197
```

2. Queue up jobs for generating figures.

```console
$ ./job_scripts/cruyff/filter_figures_plumes.sh params/nf-base.toml run afterok:12197
<redacted job id>
$ ./job_scripts/cruyff/filter_figures_params.sh params/nf-base.toml run afterok:12197
<redacted job id>
$ ./job_scripts/cruyff/filter_figures_seismic.sh params/nf-base.toml run afterok:12197
<redacted job id>
$ ./job_scripts/cruyff/filter_figures_errors.sh params/nf-base.toml run afterok:12197
<redacted job id>
```

3. When the jobs are done, I'm gonna `rsync` the figures to my laptop for viewing.

```console
$ rsync -avxS cruyff:/slimdata/gbruer3/JutulJUDIFilter/run_v4/figs .
$ xdg-open figs
```

4. Since the figures look fine, I'm gonna queue up the rest of the time steps.

```console
$ ./job_scripts/queue_multiple_steps_assim-gpu.sh params/nf-base.toml run 2 5 16 8 nothing
<redacted job id>
```

## Parallelization

`filter_transition.jl` and `filter_observe.jl` have to run a simulation on each ensemble member, independent of the other ensemble members.

But the Jutul ecosystem and JUDI are not set up to do multiple simulations in a single code file, so they will likely crash or heavily slow down after simulating many ensemble members.

So the code is set up to be parallelized through several "helper" scripts followed by a "closer" script that does the final file creation.

- Each helper script simulates ensemble member until it crashes or there are no more to simulate. The simulation results are saved in an temporary intermediate results folder.
- Once the helper scripts are complete, run the closer script to check that each ensemble member has been simulated. Then this script will move the intermediate results to the correct ensemble folder.

For example, on a slurm cluster, you could do run two helper jobs and have a closer job to finish up after the helpers are done.
```console
$ ./job_scripts/cruyff/filter_transition.sh params/base.toml run 3 helper
8368
$ ./job_scripts/cruyff/filter_transition.sh params/base.toml run 3 helper
8369
$ ./job_scripts/cruyff/filter_transition.sh params/base.toml run 3 closer afterany:8368,8369
8370
$ squeue
             JOBID PARTITION     NAME      USER ST       TIME  NODES NODELIST(REASON)
              8370       cpu   trans3   gbruer3 PD       0:00      1 (Dependency)
              8368       cpu   trans3   gbruer3  R       0:09      1 epyc0
              8369       cpu   trans3   gbruer3  R       0:07      1 epyc0
```

In slurm, job arrays are the way to submit multiple jobs as a group. This simplifies the `squeue` output and gets rid of the need to list each job ID in the dependencies.
```console
$ ./job_scripts/cruyff/filter_transition_array.sh params/base.toml run 3 helper 8
8368
$ ./job_scripts/cruyff/filter_transition.sh params/base.toml run 3 closer afterany:8368
8369
$ squeue
             JOBID PARTITION     NAME      USER ST       TIME  NODES NODELIST(REASON)
              8369       cpu   trans3   gbruer3 PD       0:00      1 (Dependency)
            8368_1       cpu   trans3   gbruer3  R       0:09      1 epyc0
            8368_2       cpu   trans3   gbruer3  R       0:07      1 epyc0
            8368_3       cpu   trans3   gbruer3  R       0:07      1 epyc0
            8368_4       cpu   trans3   gbruer3  R       0:07      1 epyc0
            8368_5       cpu   trans3   gbruer3  R       0:07      1 epyc0
            8368_6       cpu   trans3   gbruer3  R       0:07      1 epyc0
            8368_7       cpu   trans3   gbruer3  R       0:07      1 epyc0
            8368_8       cpu   trans3   gbruer3  R       0:07      1 epyc0
```

With my parameter setup, each transition job should take at most 30 minutes, each observation job should take 1 hour, and each assimilate job should take 5 minutes.
- If the transition jobs are run in batches of 4, then each transition step takes about 1 hour 5 minutes.
- If the observation jobs are run in batches of 4, then each observation step takes about 2 hour 5 minutes.
- So one step total will take about 3 hours 15 minutes.
- So five steps will take about 16 hours.

But right now, the observation jobs are limited to running one job at a time. That gives other people a chance to use the gpus.

You can change that using `scontrol`. Here's an example.
```console
$ squeue -j 9880
        JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 9880_[1-8%1]       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
$ scontrol update jobid=9880 ArrayTaskThrottle=4
$ squeue -j 9880
        JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 9880_[1-8%4]       gpu     obs1  gbruer3 PD       0:00      1 (Dependency)
```
The `%4` indicates that 4 jobs from that job array are allowed to run at the same time.

