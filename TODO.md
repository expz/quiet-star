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
- [ ] speed up training by using flash attention
- [ ] speed up training by adding hidden state generation to generate_thoughts() function
- [ ] check if there's any way to reduce memory usage

## Housekeeping
- [ ] Remove code duplication between `hidden_states()` and `generate_next_thought_token()`
- [ ] add type annotations to triton code
- [x] create abstract base class for thinking models

## Adding OpenELM
- [x] reproduce openelm model's forward pass with manually written code
- [x] convert logits without thoughts to work with openELM
- [x] convert thought generation to work with openELM
- [x] convert hidden state generation to work with openELM

## Evaluation
- [x] add generation function
- [x] add generation script
- [x] add GSM8K evaluation script
- [ ] add options to evaluate using sampling instead of greedily
- [ ] train on 512K samples and remove full disclosure if it improves performance
