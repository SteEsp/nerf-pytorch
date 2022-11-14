#!/bin/bash
# A sample script to launch trainings on all the DTU datasets

python run_nerf.py --config configs/dtu_scan55.txt
python run_nerf.py --config configs/dtu_scan63.txt
python run_nerf.py --config configs/dtu_scan69.txt
python run_nerf.py --config configs/dtu_scan83.txt
python run_nerf.py --config configs/dtu_scan97.txt
python run_nerf.py --config configs/dtu_scan105.txt
python run_nerf.py --config configs/dtu_scan106.txt
python run_nerf.py --config configs/dtu_scan110.txt
python run_nerf.py --config configs/dtu_scan114.txt
python run_nerf.py --config configs/dtu_scan122.txt


