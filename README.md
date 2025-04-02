# Graph-FM

## run for gMAE2
srun -t 50:0:0 --cpus-per-task=5 --gpus-per-node=1 python main.py --load_model 

## AWS
- Request access: https://fburl.com/HPCaaS_for_PyTorch 

## Conda environment
- Pytorch environment: https://fburl.com/wiki/f0dymta9  
- Install latest DGL, latest PyG: https://fburl.com/iacphbhb, https://fburl.com/nk7yasei
    
    ```shell script
    conda install -c dglteam/label/cu121 dgl
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
    ```
- Other packages

    ``` shell script
    pip install torchdata pandas pydantic ogb mkl
    pip install vector-quantize-pytorch
    pip install pybind11
    pip install graph-walker

    ```

## Run
- slurm: https://fburl.com/wiki/8li2klg0
    
    Example: 
    ```shell script
    srun -p train -t 5:00:00 --gpus-per-node=1 --cpus-per-task 2 python main.py 
    ```

    check status: `sinfo`, `squeue`, and `slurm_gpustat` 