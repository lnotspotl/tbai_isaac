## Distilled blind Go2 policy

```python3
policy = torch.jit.load(policy_path)
policy.set_hidden_size
...
out = policy.forward(obs).view(-1)
action = out[:12]
reconstructed = out[12:]
```