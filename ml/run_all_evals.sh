#!/bin/bash

# Run all evaluation scripts in parallel and kill all if any of them is interrupted
trap 'kill 0' SIGINT
(
    python graph_encoding/evaluate_graph_encoding.py &
    python mp_pde/evaluate_mp_pde.py &
    python rnn/evaluate_rnn.py &
    python rnn_gnn_fusion/evaluate_rnn_gnn_fusion.py &
    python repetition/evaluate_repetition.py &
    python tst/evaluate_tst.py
) 
