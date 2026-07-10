from __future__ import annotations

import json

from ml_pipeline.train import train_pipeline


if __name__ == "__main__":
    result = train_pipeline()
    print(json.dumps(result, indent=2))
