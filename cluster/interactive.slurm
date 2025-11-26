#!/bin/bash

#SBATCH --job-name "tunnel"
#SBATCH -A IscrC_TVU                      # account, shared by all the project members
#SBATCH --time 03:00:00
#SBATCH -p boost_usr_prod
#SBATCH --mem=0
#SBATCH -N 1                                # ??? 1 node, should match the number of nodes in Pl.Trainer
#SBATCH --ntasks-per-node=1                 # ??? 4 tasks out of 32, should match the number of devices in Pl.Trainer
#SBATCH --gres=gpu:1                        # 4 gpus per node out of 4
#SBATCH --error=logs/slurm/%j.err                    # standard error file
#SBATCH --output=logs/slurm/%j.out                   # standard output file

# 1. Create a directory for SSHD configurations and keys
mkdir -p ~/sshd
chmod 700 ~/sshd

# 2. Generate a host key if it doesn't exist
if [ ! -f ~/sshd/ssh_host_rsa_key ]; then
    ssh-keygen -t rsa -b 4096 -f ~/sshd/ssh_host_rsa_key -N "" -q
fi

# 3. Create SSHD configuration file
cat > ~/sshd/sshd_config << EOF
Port 2222
ListenAddress 0.0.0.0
HostKey ~/sshd/ssh_host_rsa_key
PasswordAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
AuthorizedKeysFile ~/.ssh/authorized_keys
EOF

# 4. Ensure the authorized_keys file exists
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 5. Add your public key to authorized_keys
# Replace the below line with your actual public key
cat ~/.ssh/laptop.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 6. Start SSHD with the new configuration
/usr/sbin/sshd -D -f ~/sshd/sshd_config