first 

```
pip install git+https://github.com/Jonas-Metzger/lookahead-pytorch.git
```

then just wrap another optimizer:

```
from lookahead import Lookahead, OAdam
optim = Lookahead(OAdam(model.parameters(), lr=lr))

```
