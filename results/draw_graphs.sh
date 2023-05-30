conda activate phylo
echo "doing corruption"
python corruption_results.py
echo "doing real"
python dcm_results_real.py
echo "doing dcm"
python dcm_results.py
echo "doing super"
python supertree_results.py