#!/bin/bash

# The file with the list of jobs to be sent
file="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/config/job_config_runs.txt"
path_objects="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/objects/tmp_bash/"
path_output="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/objects/output_slurm/"
# The python scripts to use
root_scripts="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/"
python_script="$root_scripts""script_scaling.py"

# Read the run numbers from the input text file
run_strs=$(cat $file)
job_ids=()

##################################################################
# Step 0: Submitting a job to compute all the IRFs id do not exist
irfs_script="$path_objects""job_irfs.sh"
echo "#!/bin/bash" > $irfs_script
echo "python $python_script 'irfs'" >> $irfs_script

# Submit the irf job
irf_job_id=$(sbatch -p short --output="$path_output""slurm_irfs-%j.out" $irfs_script | awk '{print $4}')
echo "IRF job submitted with Job ID $irf_job_id"
job_ids+=($irf_job_id)

##################################
# Step 1: Submit jobs for each run
for run_str in $run_strs; do
    # Create a temporary bash script for each run
    tmp_script_init="$path_objects""job_""$run_str""_scaling_init.sh"
    echo "#!/bin/bash" > $tmp_script_init
    echo "python $python_script 'init' '$run_str'" >> $tmp_script_init

    # Submit the job and capture the job ID
    job_id=$(sbatch -p short --output="$path_output""slurm_init-%j.out" $tmp_script_init | awk '{print $4}')
    echo "Init job submitted for run_strs: $run_str, with Job ID $job_id"

    # Store the job ID in the array
    job_ids+=($job_id)
done

##########################################################################
# Step 2: Submit a job to merge information after all previous jobs finish
merge_script="$path_objects""job_merge.sh"
echo "#!/bin/bash" > $merge_script
echo "python $python_script 'merge'" >> $merge_script

# Get the job IDs of all previously submitted jobs join by ":"
delimiter=":"
dependency_runs="${job_ids[*]:-}" 
dependency_runs="${dependency_runs//$' '/$delimiter}"

# Submit the merge job with dependencies
merge_job_id=$(sbatch -p short --output="$path_output""slurm_merge-%j.out" --dependency=afterok:$dependency_runs $merge_script | awk '{print $4}')
echo "Merge job submitted with Job ID $merge_job_id"


##############################################################################################
# Step 3: Submit jobs for each run with a different python script after the merge job finishes
for run_str in $run_strs; do
    # Create a temporary bash script for each run after merge
    tmp_script_final="$path_objects""job_""$run_str""_scaling_final.sh"
    echo "#!/bin/bash" > $tmp_script_final
    echo "python $python_script 'final' '$run_str'" >> $tmp_script_final

    # Submit the job with dependency on the merge job
    job_id=$(sbatch -p short --output="$path_output""slurm_final-%j.out" --dependency=afterok:$merge_job_id $tmp_script_final | awk '{print $4}')
    echo "Final job submitted for run $run_str, with Job ID $job_id"
done
