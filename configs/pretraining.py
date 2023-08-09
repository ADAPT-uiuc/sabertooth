# Copyright 2020 The Sabertooth Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_config(config_string="base"):
    ## Type of attention mechanism.
    attn_type = "ReformerMHA"
    ## Downsampling factor 
    downsampling_k = 64
    ## Dropout for ffn
    ffn_dropout = 0.1
    ## Up_train flag for a 70-30 split
    up_train = True
    if config_string == "large":
        model_config = ml_collections.ConfigDict(
            {
                "vocab_size": 32128,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "hidden_act": "gelu_new",
                "intermediate_size": 4096,
                "hidden_dropout_prob": ffn_dropout,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "attention_type": attn_type,
                "downsampling_k": downsampling_k,
                "up_train": up_train,
                "use_t5_rpe": True,
                "window_size": 16,
                "num_landmarks": 16,
                "overlap_window": False
            }
        )
    else:
        assert config_string == "base" or not config_string
        model_config = ml_collections.ConfigDict(
            {
                "vocab_size": 32128,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "hidden_act": "gelu_new",
                "intermediate_size": 3072,
                "hidden_dropout_prob": ffn_dropout,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "attention_type": attn_type,
                "downsampling_k": downsampling_k,
                "up_train": up_train,
                "use_t5_rpe": True,
                "window_size": 16,
                "num_landmarks": 16,
                "overlap_window": False
            }
        )

    config = ml_collections.ConfigDict(
        {
            # Configuration for the model
            "model": model_config,
            # Initial checkpoint
            "init_checkpoint": "",
            # Input files
            "input_files": ["/srv/local/shared/pre-train-mixture/wikibooks/part-*-of-00500.jsonl"],
            # Pre-trained tokenizer
            "tokenizer": "/srv/local/shared/pre-train-mixture/wikibooks_32k.model",
            # Whether to run training.
            "do_train": True, 
            # Whether to run eval.
            "do_eval": True,
            # Total batch size for training.
            "train_batch_size": 1024,
            # Total batch size for eval.
            "eval_batch_size": 64,
            # Optimizer: either 'adam' or 'lamb
            "optimizer": "adam",
            # The base learning rate for Adam or LAMB.
            "learning_rate": 1.7e-4, 
            # The beta1 parameter for Adam or LAMB
            "adam_beta1": 0.9,
            # The beta2 parameter for Adam or LAMB
            "adam_beta2": 0.999,
            # The epsilon parameter for Adam or LAMB
            "adam_epsilon": 1e-6,
            # Weight decay rate for all parameters except biases and layer norm scale
            "weight_decay": 0.01,
            # Maximum gradient norm (for gradient clipping)
            "max_grad_norm": 1.0,
            # Number of training steps.
            "num_train_steps": 120000,
            # Number of warmup steps.
            "num_warmup_steps": 10000,
            # The maximum total input sequence length after tokenization.
            # Sequences longer than this will be truncated, and sequences shorter
            # than this will be padded.
            "max_seq_length": 128,
            # Maximum number of masked LM predictions per sequence.
            "max_predictions_per_seq": 20,
            # How often to save the model checkpoint.
            "save_checkpoints_steps": 10000,
            # Maximum number of eval steps.
            "max_eval_steps": 1000,
        }
    )

    return config
