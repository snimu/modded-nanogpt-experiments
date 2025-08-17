# 00004-x0

Experiments with the x0 parameter.

## 2025-08-17-x00-x01

The idea is to add a second embedding-layer (ignoring the value embeddings). Then, instead of doing this:

```python
x = x0 = norm(self.embed(input_tokens))
for layer in layer:
    x = x_lambda * x + x0_lambda * x0
    x = block(x)  # ignoring all the other shenanigangs
```

We do this:

```python
x = x00 = norm(self.embed0(input_tokens))
x01 = norm(self.embed1(input_tokens))
for layer in layer:
    x = x_lambda * x + x00_lambda * x0 + x01_lambda * x01
    x = block(x)  # ignoring all the other shenanigangs
```

I think it's unlikely that this will actually help set a new record (because it will make every step slower, and would thus have to make learning a lot easier for the model), but I'm curious to at least try it out for these reasons:

- Value emeddings work. They're pretty similar, but are added in a different manner. So cheap additional parameters via embeddings seem to be a good thing.
- Adding in x0 helps. But why would the model benefit as strongly from x0 at layer 15 as it does at layer 0?
- Having two different embeddings that are added in a weighted sum to the residual at every layer, instead of one of them, will allow the model to learn to make use of one of the embeddings more in some layers, and the other in other layers

Of course, if this does *anything*, I should try just adding more and more x0s, just to see what happens.
