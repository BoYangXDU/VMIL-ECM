"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        default="simulated-data",
        type=str,
        help='data path [gulfport-data   simulated-data]'
    )

    parser.add_argument(
        '--target_type',
        default="0.3",
        type=str,
        help='target types[0.1   0.2   0.3  ]'
    )

    parser.add_argument(
        '--_fc_dim',
        default=1536,
        type=int,
        help='fc dim [384 for gulfport, and 1536 for simulated data]'
    )

    parser.add_argument(
        '--_spectral_dim',
        default=211,
        type=int,
        help='spectral dim [64 for gulfport, and 211 for simulated data]'
    )


    parser.add_argument(
        '--learning_rate',
        default=1 * 1e-3,
        type=float,
        help='default initial learning rate (later will change in main.py)'
    )

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='batch size for training'
    )

    parser.add_argument(
        '--_gamma',
        default=0.15,
        type=float,
        help='hyper parameter gamma'
    )

    parser.add_argument(
        '--_nlayer',
        default=5,
        type=float,
        help='number of transformer layers'
    )

    parser.add_argument(
        '--manual_seed',
        default=1,
        type=int,
        help='random seed'
    )

    parser.add_argument(
        '--p_epoches',
        default=500,
        type=int,
        help='pretrain epoches'
    )

    parser.add_argument(
        '--t_epoches',
        default=2000,
        type=int,
        help='totol training epoches'
    )

    args = parser.parse_args()
    return args