#!/bin/bash

# The file with the list of jobs to be sent
file="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/config/job_config_runs.txt"

# Counter for the number of jobs in a group
counter=0

# read the lines of the given file
while read -r line; do
  # only operating if the line is not commented with #
  if [[ "${line:0:1}" != '#' ]]; then
    str="$line"

    # Create a new temporary job script if the counter is a multiple of 3
    if (( counter % 3 == 0 )); then
      echo "#! /bin/bash" > "bash_dl1a_to_dl1b_tmpjob.sh"
    fi

    echo "python /fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/dl1a_to_dl1b_scaling.py '$str'" >> "bash_dl1a_to_dl1b_tmpjob.sh"

    # Increment the counter
    ((counter++))
    echo "Adding $str to tmp job"

    # Check if the counter is 3
    if (( counter % 3 == 0 )); then
      echo -e "Sending tmp job group to the queue...\n"
      sbatch -p long --output="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/objects/output_slurm/slurm-%j.out" "bash_dl1a_to_dl1b_tmpjob_$((counter/3)).sh"
      counter=0
    fi
  fi

done < "$file"

# Check if the counter is 1 or 2
if (( counter == 1 || counter == 2 )); then
  echo -e "Sending job group to the queue...\n"
  sbatch -p long --output="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/objects/output_slurm/slurm-%j.out" "bash_dl1a_to_dl1b_tmpjob_$((counter/3)).sh"
fi
