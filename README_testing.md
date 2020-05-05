# Testing Process

Testing is all automated, test parse automatically new files in the repo 
and add it to the testing process


### Overview

#### Configuration of auto-test :
https://github.com/arita37/mlmodels/blob/dev/.github/workflows/

    test_cli         : CLI
    test_dataloader  : dataloader part
    test_benchmark   : benchmark
    test_fast_linux  : On Commit
    test_pullrequest : On PR




####  Code of auto-test :
https://github.com/arita37/mlmodels/blob/dev/mlmodels/ztest.py




#### Log of testing
Uploaded directly here :
https://github.com/arita37/mlmodels_store





####  Helper code
```python
     from mlmodels.util import os_package_root_path, log, 
                            path_norm, get_model_uri, path_norm_dict

     ### Use path_norm to normalize your path.
     data_path = path_norm("dataset/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/dataset/text/myfile.txt


     ### Use path_norm to normalize your path.
     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt


     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt
```





