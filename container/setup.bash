#!/usr/bin/env bash

HOME=/home/tbai
source $HOME/.bashrc

echo "Installing tbai_bindings..."
cd $HOME/tbai_bindings && source devel/setup.bash
rm $HOME/tbai_isaac/src/tbai_isaac/anymal_d/dtc/tbai_ocs2_interface.cpython-38-x86_64-linux-gnu.so
cd $HOME/tbai_bindings/src/tbai_bindings/dependencies && rm libtorch && ln -s ${HOME}/libtorch .
catkin build tbai_bindings

echo "Creating symlink for tbai_ocs2_interface..."
ln -s $(pwd)/devel/lib/tbai_ocs2_interface.cpython-38-x86_64-linux-gnu.so $HOME/tbai_isaac/src/tbai_isaac/anymal_d/dtc

echo "Installing tbai_isaac..."
cd $HOME/tbai_bindings && source devel/setup.bash
cd $HOME/tbai_isaac && pip3 install -e .

export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\] (container):\[\033[01;34m\]\w\[\033[00m\]\$ '

echo "All done. Enjoy 🤗"