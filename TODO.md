# TODO

## Basic algorithm
- [x] implement logits without thoughts
- [x] implement mixing head
- [x] implement logits mixing
- [x] implement computation of negative log likelihood
- [x] implement computation of reward
- [x] implement computation of policy gradient
- [x] implement overall loss
- [x] add gradient weighting

## Using a pretrained model
- [x] convert MLX code to PyTorch
- [x] reproduce pretrained model's forward pass with manually written code
- [x] convert logits without thoughts to work with pretrained Huggingface model
- [x] convert thought generation to work with pretrained Huggingface model
- [x] convert hidden state generation to work with pretrained Huggingface model
- [x] convert LM head to work with pretrained Huggingface model
- [x] convert mixing head to work with pretrained Huggingface model

## Housekeeping
- [ ] add type annotations to triton code
- [ ] create abstract base class for thinking models
