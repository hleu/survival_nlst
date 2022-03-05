To install packages needed for the code, use the requirements in ``env_reqs.txt`` in the ``media_code`` dir. You can use ``pip install -r env_reqs.txt`` in a virtual environment. 

To generate the final results, run the code ``generate_all_results.py`` in ``media_code`` which uses ``results_utils.py`` and ``surv_model_utils.py``. To run the code and generate a ``results.txt`` file, use the code ``python generate_all_results.py | tee results.txt`` or directly use the bash file ``run_code.sh`` 
