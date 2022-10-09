first 

```
pip install git+https://github.com/Jonas-Metzger/lookahead-pytorch.git
```

then 

```
from lookahead import Lookahead
```

and then just wrap another optimizer, like this:

```
optim = Lookahead(torch.optim.Adam(model.parameters(), lr=lr))
```

