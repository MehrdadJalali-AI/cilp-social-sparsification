#!/bin/bash
# Final-phase lane for one dataset, in priority order. Usage: lane.sh <dataset> <threads>
ds=$1; th=$2; R="python3 -W ignore scripts/pipeline/run_experiment.py --phase final --datasets $ds --threads $th"
LOG=results/main/logs/final_$ds.log
$R --variants cilp_full task_only >> $LOG 2>&1
python3 -W ignore scripts/pipeline/run_baselines.py --datasets $ds --threads $th >> $LOG 2>&1
$R --variants loo_task loo_comm loo_conn loo_spec loo_repr loo_group grp_deletion_only grp_task_plus_proxies grp_proxies_only >> $LOG 2>&1
$R --variants w_balanced w_task_focused w_structure_focused w_reduced_proxy >> $LOG 2>&1
$R >> $LOG 2>&1   # remaining Dirichlet variants
$R --variants cilp_full task_only --n 20 40 80 120 240 >> $LOG 2>&1
if [ "$ds" != "pubmed" ]; then $R --variants cilp_full task_only --stage-a-setting >> $LOG 2>&1; fi
echo "LANE DONE $ds" >> $LOG
