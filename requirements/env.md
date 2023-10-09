conda create -n trustenv python=3.10
conda activate trustenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge opencv
conda install numpy
conda install -c anaconda scikit-learn