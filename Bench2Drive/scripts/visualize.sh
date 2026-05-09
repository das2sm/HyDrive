export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/visualization/visualize.py \
	projects/configs/sparsedrive_stage2.py \
	--num-workers 1 \
	--start 0 \
	--end 10 \
	--interval 1 \
	--result-path work_dirs/sparsedrive_stage2/results.pkl