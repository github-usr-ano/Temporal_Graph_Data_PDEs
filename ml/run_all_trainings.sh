#!/bin/bash

# Run all training scripts in parallel and kill all if any of them is interrupted
trap 'kill 0' SIGINT
(
    python graph_encoding/train_graph_encoding.py &
    python mp_pde/train_mp_pde.py &
    python rnn/train_rnn.py &
    python rnn_gnn_fusion/train_rnn_gnn_fusion.py &
    python tst/train_tst.py
) 
