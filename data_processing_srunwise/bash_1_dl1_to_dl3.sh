#!/bin/bash

# The file with the list of jobs to be sent
file="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/config/job_config_runs.txt"

# Output and temporal slurm bash launchers
path_objects="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/objects/tmp_bash/"
path_output="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/objects/output_slurm/"

# The python scripts to use
root_scripts="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing_srunwise/"
python_script="$root_scripts""script_2_dl1_to_dl3.py"

# Read the run numbers from the input text file
input_strs=$(cat $file)


# Create an array to store the runs
runs=()

# Iterate over each input_str
while IFS= read -r input_str; do
    # Extract the run name using parameter expansion
    run_name="${input_str%%_*}"
    # Add the run name to the array
    runs+=("$run_name")
done <<< "$input_strs"

unique_runs=($(printf "%s\n" "${runs[@]}" | sort -u))

for run_str in "${unique_runs[@]}"; do
    # Create a temporary bash script for each run after merge
    tmp_script_final_scaled="$path_objects""job_dl1_to_dl3_scaled_""$run_str"".sh"
    tmp_script_final="$path_objects""job_dl1_to_dl3_""$run_str"".sh"
    echo "#!/bin/bash" > $tmp_script_final_scaled
    echo "#!/bin/bash" > $tmp_script_final
    echo "python $python_script '$run_str' 'True'" >> $tmp_script_final_scaled
    echo "python $python_script '$run_str' 'False'" >> $tmp_script_final

    # Submit the job with dependency on the merge job
    job_id=$(sbatch -p short --mem=80000 --output="$path_output""slurm_dl1_to_dl3_scaled-%j.out" $tmp_script_final_scaled | awk '{print $4}')
    echo "Job for scaled data submitted for run $run_str, with Job ID $job_id"
    job_id=$(sbatch -p short --mem=80000 --output="$path_output""slurm_dl1_to_dl3_original-%j.out" $tmp_script_final | awk '{print $4}')
    echo "Job for original data submitted for run $run_str, with Job ID $job_id"

    
done

