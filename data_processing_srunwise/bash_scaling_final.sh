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

##############################################################################################
# Step 3: Submit jobs for each run with a different python script after the merge job finishes
for run_str in $run_strs; do
    # Create a temporary bash script for each run after merge
    tmp_script_final="$path_objects""job_""$run_str""_scaling_final.sh"
    echo "#!/bin/bash" > $tmp_script_final
    echo "python $python_script 'final' '$run_str'" >> $tmp_script_final

    # Submit the job with dependency on the merge job
    job_id=$(sbatch -p short --output="$path_output""slurm_final-%j.out" $tmp_script_final | awk '{print $4}')
    echo "Final job submitted for run $run_str, with Job ID $job_id"
done
