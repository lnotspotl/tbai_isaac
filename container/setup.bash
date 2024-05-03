#!/usr/bin/env bash

HOME=/home/tbai
source $HOME/.bashrc

echo "Installing tbai_bindings..."
cd $HOME/tbai_bindings && source devel/setup.bash
catkin build tbai_bindings

echo "Creating symlink for ig_interface..."
rm $HOME/tbai_isaac/src/tbai_isaac/anymal_d/dtc/ig_interface.cpython-38-x86_64-linux-gnu.so
ln -s $(pwd)/devel/lib/ig_interface.cpython-38-x86_64-linux-gnu.so $HOME/tbai_isaac/src/tbai_isaac/anymal_d/dtc

echo "Installing tbai_isaac..."
cd $HOME/tbai_bindings && source devel/setup.bash
cd $HOME/tbai_isaac && pip3 install -e .

echo "All done. Enjoy 🤗"