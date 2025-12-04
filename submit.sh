#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpul40s
### -- set the job Name --
#BSUB -J 4monthgrid
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


### Activate venv 
module load python3/3.10.12
module load cuda/12.8.0
REPO="/dtu/blackhole/0e/213582/dark-vessel-hunter/"
cd $REPO
source dlproject/bin/activate 

### Here follow the commands you want to execute with input.in as the input file
python training_run.py