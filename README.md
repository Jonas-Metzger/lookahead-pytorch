first 

```
pip install git+https://github.com/Jonas-Metzger/lookahead-pytorch.git
```

then just wrap another optimizer:

```
from lookahead import Lookahead, OAdam
optim = Lookahead(OAdam(model.parameters(), lr=lr))

```

during your training loop, insert

```
if step % 3 == 0: optim.pullback()
```
