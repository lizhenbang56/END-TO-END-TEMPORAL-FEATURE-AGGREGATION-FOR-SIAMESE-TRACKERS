# source activate pytorch1.4_python3.7  # etvuz@172.18.32.31
source activate py37-pytorch1.4  # yyshi@172.18.32.157
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python setup.py build develop