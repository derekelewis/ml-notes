#!/bin/bash
su - ec2-user -c 'bash -s' <<'EOF'
    sudo dnf install -y unzip
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    RH_ACTIVATION_KEY=$(aws ssm get-parameter --name /rh-activation-key-name --query "Parameter.Value" --output text --with-decryption)
    RH_ACTIVATION_ORG=$(aws ssm get-parameter --name /rh-activation-key-org --query "Parameter.Value" --output text --with-decryption)
    sudo subscription-manager register --activationkey=${RH_ACTIVATION_KEY} --org=${RH_ACTIVATION_ORG}
    sudo yum update -y
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
    bash ./Miniconda3-py311_23.5.2-0-Linux-x86_64.sh -b
    miniconda3/bin/conda init bash
    source .bashrc
    conda create -yn mlenv
    conda activate mlenv
    echo "conda activate mlenv" >> .bashrc
    sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    sudo dnf -y module install nvidia-driver:latest-dkms
    sudo dnf -y install cuda
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 matplotlib jupyterlab -c pytorch -c nvidia
    pip install watermark
EOF